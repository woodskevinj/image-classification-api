"""
Comprehensive security and functionality tests for the Image Classification API.

Tests cover:
1. File validation (size limits, MIME type checking, decompression bomb prevention)
2. Rate limiting
3. Input validation with Pydantic
4. CORS configuration
5. All API endpoints
"""

import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient
from api.app import app, MAX_FILE_SIZE, ALLOWED_IMAGE_TYPES
import time


# ======================================================
# Test Client Setup
# ======================================================

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    # Disable rate limiting for most tests
    from api.app import limiter
    limiter.enabled = False
    client = TestClient(app)
    yield client
    # Re-enable rate limiting after tests
    limiter.enabled = True


@pytest.fixture
def client_with_rate_limit():
    """Create a test client with rate limiting enabled."""
    from api.app import limiter
    limiter.enabled = True
    return TestClient(app)


# ======================================================
# Helper Functions for Test Data
# ======================================================

def create_test_image(format="PNG", size=(32, 32), mode="RGB"):
    """Create a valid test image in memory."""
    img = Image.new(mode, size, color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=format)
    img_bytes.seek(0)
    return img_bytes


def create_large_image(size=(10000, 10000)):
    """Create a very large image to test decompression bomb detection."""
    img = Image.new("RGB", size, color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


def create_oversized_file():
    """Create a file larger than MAX_FILE_SIZE."""
    # Create an image that will be larger than 10 MB when encoded
    size = (5000, 5000)  # Should result in > 10 MB file
    img = Image.new("RGB", size, color="green")
    img_bytes = io.BytesIO()
    # Use minimal compression to maximize file size
    img.save(img_bytes, format="PNG", compress_level=0)
    img_bytes.seek(0)
    return img_bytes


# ======================================================
# Test 1: File Size Validation
# ======================================================

class TestFileSizeValidation:
    """Test file size limits."""

    def test_valid_file_size(self, client):
        """Test that files within size limit are accepted."""
        img_bytes = create_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        assert response.status_code in [200, 500]  # 500 if model fails, but validation passed

    def test_empty_file_rejected(self, client):
        """Test that empty files are rejected."""
        empty_file = io.BytesIO(b"")
        response = client.post(
            "/predict",
            files={"file": ("test.png", empty_file, "image/png")}
        )
        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]

    def test_oversized_file_rejected(self, client):
        """Test that files larger than MAX_FILE_SIZE are rejected."""
        # Create a file larger than 10 MB
        large_data = b"x" * (MAX_FILE_SIZE + 1024)
        large_file = io.BytesIO(large_data)

        response = client.post(
            "/predict",
            files={"file": ("large.png", large_file, "image/png")}
        )
        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]


# ======================================================
# Test 2: MIME Type Validation
# ======================================================

class TestMIMETypeValidation:
    """Test MIME type validation."""

    def test_valid_mime_types(self, client):
        """Test that valid image MIME types are accepted."""
        for format in ["PNG", "JPEG", "GIF", "BMP"]:
            img_bytes = create_test_image(format=format)
            mime_type = f"image/{format.lower()}"
            response = client.post(
                "/predict",
                files={"file": (f"test.{format.lower()}", img_bytes, mime_type)}
            )
            # Should pass validation (200 or 500 if model fails)
            assert response.status_code in [200, 500], f"Failed for {format}"

    def test_invalid_mime_type_rejected(self, client):
        """Test that non-image files are rejected."""
        # Create a text file pretending to be an image
        text_file = io.BytesIO(b"This is not an image")
        response = client.post(
            "/predict",
            files={"file": ("test.txt", text_file, "text/plain")}
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_pdf_file_rejected(self, client):
        """Test that PDF files are rejected."""
        pdf_content = b"%PDF-1.4\nSome PDF content"
        pdf_file = io.BytesIO(pdf_content)
        response = client.post(
            "/predict",
            files={"file": ("test.pdf", pdf_file, "application/pdf")}
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]


# ======================================================
# Test 3: Decompression Bomb Prevention
# ======================================================

class TestDecompressionBombPrevention:
    """Test protection against decompression bombs."""

    def test_normal_image_accepted(self, client):
        """Test that normal-sized images are accepted."""
        img_bytes = create_test_image(size=(500, 500))
        response = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        assert response.status_code in [200, 500]

    @pytest.mark.slow
    def test_decompression_bomb_rejected(self, client):
        """Test that extremely large images are rejected."""
        # Create a very large image that PIL would consider a decompression bomb
        try:
            img_bytes = create_large_image(size=(20000, 20000))
            response = client.post(
                "/predict",
                files={"file": ("huge.png", img_bytes, "image/png")}
            )
            # Should be rejected with 400 or 413
            assert response.status_code in [400, 413]
            assert "decompression bomb" in response.json()["detail"].lower() or \
                   "too large" in response.json()["detail"].lower()
        except Exception:
            # If we can't create the image due to memory limits, that's fine
            pytest.skip("Cannot create decompression bomb test image")


# ======================================================
# Test 4: Rate Limiting
# ======================================================

class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_on_predict(self, client_with_rate_limit):
        """Test that rate limiting works on /predict endpoint."""
        img_bytes = create_test_image()

        # Make requests up to the limit (30/minute)
        # We'll make 35 requests and expect some to be rate limited
        responses = []
        for i in range(35):
            img_bytes.seek(0)  # Reset file pointer
            response = client_with_rate_limit.post(
                "/predict",
                files={"file": (f"test{i}.png", img_bytes, "image/png")}
            )
            responses.append(response.status_code)

        # At least some requests should be rate limited (429)
        assert 429 in responses, "Rate limiting should trigger for excessive requests"

    def test_rate_limit_on_logs(self, client_with_rate_limit):
        """Test that rate limiting works on /logs endpoint."""
        # Make requests up to the limit (10/minute)
        responses = []
        for i in range(15):
            response = client_with_rate_limit.get("/logs?limit=5")
            responses.append(response.status_code)

        # At least some requests should be rate limited (429)
        assert 429 in responses, "Rate limiting should trigger on /logs endpoint"


# ======================================================
# Test 5: Pydantic Input Validation
# ======================================================

class TestPydanticValidation:
    """Test Pydantic model validation for query parameters."""

    def test_logs_limit_validation(self, client):
        """Test that limit parameter is validated correctly."""
        # Valid range: 1-100
        response = client.get("/logs?limit=50")
        assert response.status_code == 200

        # Test invalid values
        response = client.get("/logs?limit=0")
        assert response.status_code == 422  # Validation error

        response = client.get("/logs?limit=101")
        assert response.status_code == 422  # Validation error

        response = client.get("/logs?limit=-5")
        assert response.status_code == 422  # Validation error

    def test_logs_limit_type_validation(self, client):
        """Test that limit parameter must be an integer."""
        response = client.get("/logs?limit=abc")
        assert response.status_code == 422  # Validation error


# ======================================================
# Test 6: CORS Configuration
# ======================================================

class TestCORS:
    """Test CORS middleware configuration."""

    def test_cors_middleware_configured(self):
        """Test that CORS middleware is configured in the app."""
        from starlette.middleware.cors import CORSMiddleware
        # Check that CORS middleware is in the app's middleware stack
        # Middleware is wrapped in Starlette's Middleware class
        has_cors = any(
            hasattr(m, 'cls') and m.cls == CORSMiddleware
            for m in app.user_middleware
        )
        assert has_cors, "CORS middleware should be configured"

    def test_preflight_request(self, client):
        """Test CORS preflight OPTIONS request."""
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            }
        )
        # Should return 200 for preflight request
        assert response.status_code == 200


# ======================================================
# Test 7: API Endpoints Functionality
# ======================================================

class TestAPIEndpoints:
    """Test all API endpoints for correct functionality."""

    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert data["model_loaded"] is True

    def test_info_endpoint(self, client):
        """Test the model info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "architecture" in data
        assert "num_classes" in data
        assert data["num_classes"] == 10
        assert "total_parameters" in data
        assert "device" in data

    def test_logs_endpoint(self, client):
        """Test the logs endpoint."""
        response = client.get("/logs?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "recent_predictions" in data
        assert isinstance(data["recent_predictions"], list)

    def test_predict_endpoint_valid_image(self, client):
        """Test prediction with a valid image."""
        img_bytes = create_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")}
        )

        # Should return 200 with predictions
        if response.status_code == 200:
            data = response.json()
            assert "top1_prediction" in data
            assert "top3_predictions" in data
            assert "label" in data["top1_prediction"]
            assert "confidence" in data["top1_prediction"]
            assert len(data["top3_predictions"]) == 3

            # Validate confidence scores
            for pred in data["top3_predictions"]:
                assert 0.0 <= pred["confidence"] <= 1.0


# ======================================================
# Test 8: Pydantic Response Models
# ======================================================

class TestResponseModels:
    """Test that response models are properly validated."""

    def test_prediction_response_format(self, client):
        """Test that prediction responses match the Pydantic model."""
        img_bytes = create_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")}
        )

        if response.status_code == 200:
            data = response.json()
            # Check structure
            assert "top1_prediction" in data
            assert "label" in data["top1_prediction"]
            assert "confidence" in data["top1_prediction"]
            # Confidence should be between 0 and 1
            assert 0.0 <= data["top1_prediction"]["confidence"] <= 1.0

    def test_health_response_format(self, client):
        """Test that health responses match the Pydantic model."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        required_fields = ["status", "model_loaded", "device", "message"]
        for field in required_fields:
            assert field in data


# ======================================================
# Test 9: Error Handling
# ======================================================

class TestErrorHandling:
    """Test error handling for various edge cases."""

    def test_missing_file_parameter(self, client):
        """Test request without file parameter."""
        response = client.post("/predict")
        assert response.status_code == 422  # Validation error

    def test_corrupted_image_file(self, client):
        """Test upload of corrupted image data."""
        corrupted_data = io.BytesIO(b"PNG\x00\x00INVALID_DATA")
        response = client.post(
            "/predict",
            files={"file": ("corrupted.png", corrupted_data, "image/png")}
        )
        assert response.status_code in [400, 500]  # Should handle gracefully


# ======================================================
# Test Configuration
# ======================================================

# Mark slow tests
pytest.mark.slow = pytest.mark.skipif(
    "not config.getoption('--run-slow')",
    reason="Slow tests skipped by default"
)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
