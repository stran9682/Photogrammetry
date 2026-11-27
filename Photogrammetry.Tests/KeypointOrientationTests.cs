using PhotogrammetryLibrary;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Photogrammetry.Tests;

public class KeypointOrientationTests
{
	private static double AngularDifference(double a, double b)
	{
		var diff = Math.Abs(a - b) % 360.0;
		return diff > 180.0 ? 360.0 - diff : diff;
	}

	private static Image<L8> CreateRampImage(int width, int height, double angleDegrees, double centerCol,
		double centerRow)
	{
		var angleRad = angleDegrees * Math.PI / 180.0;
		var cosA = Math.Cos(angleRad);
		var sinA = Math.Sin(angleRad);

		var image = new Image<L8>(width, height);
		image.ProcessPixelRows(accessor =>
		{
			for (var r = 0; r < accessor.Height; r++)
			{
				var span = accessor.GetRowSpan(r);
				for (var c = 0; c < span.Length; c++)
				{
					var proj = (c - centerCol) * cosA + (r - centerRow) * sinA;
					var val = Math.Clamp(128.0 + 2.0 * proj, 0.0, 255.0);
					span[c] = new L8((byte)Math.Round(val));
				}
			}
		});

		return image;
	}

	[Fact]
	public void AssignOrientation_HorizontalGradientEast_AssignsZeroOr360Degrees()
	{
		// Brightness increases strictly from left to right (dx > 0, dy = 0)
		using var image = CreateRampImage(40, 40, 0, 20, 20);
		var coordinate = new Coordinate(2, 20, 20);

		var keypoints = Sift.AssignOrientation(coordinate, image);

		Assert.NotEmpty(keypoints);
		var kp = Assert.Single(keypoints);
		Assert.True(AngularDifference(kp.Orientation, 0.0) < 5.0,
			$"Expected orientation ~0°/360°, got {kp.Orientation}°");
	}

	[Fact]
	public void AssignOrientation_HorizontalGradientWest_Assigns180Degrees()
	{
		// Brightness increases strictly from right to left (dx < 0, dy = 0)
		using var image = CreateRampImage(40, 40, 180, 20, 20);
		var coordinate = new Coordinate(2, 20, 20);

		var keypoints = Sift.AssignOrientation(coordinate, image);

		Assert.NotEmpty(keypoints);
		var kp = Assert.Single(keypoints);
		Assert.True(AngularDifference(kp.Orientation, 180.0) < 5.0,
			$"Expected orientation ~180°, got {kp.Orientation}°");
	}

	[Fact]
	public void AssignOrientation_VerticalGradientSouth_Assigns270Degrees()
	{
		// Brightness increases strictly downwards (dx = 0, dy > 0 in screen space)
		// 360 - 90 = 270 degrees in counter-clockwise Cartesian convention
		using var image = CreateRampImage(40, 40, 90, 20, 20);
		var coordinate = new Coordinate(2, 20, 20);

		var keypoints = Sift.AssignOrientation(coordinate, image);

		Assert.NotEmpty(keypoints);
		var kp = Assert.Single(keypoints);
		Assert.True(AngularDifference(kp.Orientation, 270.0) < 5.0,
			$"Expected orientation ~270°, got {kp.Orientation}°");
	}

	[Fact]
	public void AssignOrientation_VerticalGradientNorth_Assigns90Degrees()
	{
		// Brightness increases strictly upwards (dx = 0, dy < 0 in screen space)
		// 360 - 270 = 90 degrees
		using var image = CreateRampImage(40, 40, 270, 20, 20);
		var coordinate = new Coordinate(2, 20, 20);

		var keypoints = Sift.AssignOrientation(coordinate, image);

		Assert.NotEmpty(keypoints);
		var kp = Assert.Single(keypoints);
		Assert.True(AngularDifference(kp.Orientation, 90.0) < 5.0,
			$"Expected orientation ~90°, got {kp.Orientation}°");
	}

	[Theory]
	[InlineData(30.0)]
	[InlineData(45.0)]
	[InlineData(120.0)]
	[InlineData(215.0)]
	public void AssignOrientation_ArbitraryGradientAngles_InterpolatesAccurately(double screenAngleDegrees)
	{
		using var image = CreateRampImage(50, 50, screenAngleDegrees, 25, 25);
		var coordinate = new Coordinate(2, 20, 20);

		var expectedCartesian = (360.0 - screenAngleDegrees) % 360.0;
		var keypoints = Sift.AssignOrientation(coordinate, image);

		Assert.NotEmpty(keypoints);
		var kp = Assert.Single(keypoints);
		// Histogram bin width is 10° (360° / 36), so bin quantization error is at most 5°.
		Assert.True(AngularDifference(kp.Orientation, expectedCartesian) <= 5.5,
			$"Screen angle {screenAngleDegrees}°: expected {expectedCartesian}°, got {kp.Orientation}°");
	}

	[Fact]
	public void AssignOrientation_UniformImage_AssignsFallbackOrientationZero()
	{
		using var image = new Image<L8>(30, 30);
		image.ProcessPixelRows(accessor =>
		{
			for (var r = 0; r < accessor.Height; r++)
			{
				var span = accessor.GetRowSpan(r);
				for (var c = 0; c < span.Length; c++) span[c] = new L8(128);
			}
		});

		var coordinate = new Coordinate(2, 15, 15);
		var keypoints = Sift.AssignOrientation(coordinate, image);

		Assert.Single(keypoints);
		Assert.Equal(0.0, keypoints[0].Orientation);
		Assert.Equal(15, keypoints[0].X);
		Assert.Equal(15, keypoints[0].Y);
		Assert.Equal(2, keypoints[0].S);
	}

	[Fact]
	public void AssignOrientation_BimodalGradient_ProducesMultipleKeypointsWithin80PercentPeak()
	{
		// Create an image with two distinct directional halves:
		// Left side has strong horizontal gradient (0°), right side has strong vertical gradient (90°)
		using var image = new Image<L8>(50, 50);
		image.ProcessPixelRows(accessor =>
		{
			for (var r = 0; r < accessor.Height; r++)
			{
				var span = accessor.GetRowSpan(r);
				for (var c = 0; c < span.Length; c++)
					if (c < 25)
					{
						// Horizontal ramp
						var val = Math.Clamp(128.0 + 3.0 * (c - 25), 0.0, 255.0);
						span[c] = new L8((byte)val);
					}
					else
					{
						// Vertical ramp
						var val = Math.Clamp(128.0 + 3.0 * (r - 25), 0.0, 255.0);
						span[c] = new L8((byte)val);
					}
			}
		});

		var coordinate = new Coordinate(2, 25, 25);
		var keypoints = Sift.AssignOrientation(coordinate, image);

		// A bimodal distribution with two distinct prominent peaks creates multiple keypoints at the same coordinate
		Assert.True(keypoints.Count >= 2,
			$"Expected at least 2 keypoints for bimodal distribution, but got {keypoints.Count}");

		// All keypoints must share the same (X, Y, S) location
		Assert.All(keypoints, kp =>
		{
			Assert.Equal(25, kp.X);
			Assert.Equal(25, kp.Y);
			Assert.Equal(2, kp.S);
		});

		// One peak corresponds to ~0°/360° and another corresponds to ~270°
		Assert.Contains(keypoints, kp => AngularDifference(kp.Orientation, 0.0) < 20.0);
		Assert.Contains(keypoints, kp => AngularDifference(kp.Orientation, 270.0) < 20.0);
	}

	[Fact]
	public void AssignOrientation_BoundaryKeypoint_DoesNotThrowAndAssignsValidOrientation()
	{
		// Keypoint near the top-left boundary where window clips against image boundaries
		using var image = CreateRampImage(30, 30, 0, 3, 3);
		var coordinate = new Coordinate(2, 3, 3);

		var keypoints = Sift.AssignOrientation(coordinate, image);

		Assert.NotEmpty(keypoints);
		Assert.All(keypoints, kp =>
		{
			Assert.True(kp.Orientation >= 0.0 && kp.Orientation <= 360.0);
			Assert.Equal(3, kp.X);
			Assert.Equal(3, kp.Y);
		});
	}

	[Fact]
	public void AssignOrientation_GaussianDistanceWeighting_FavorsInnerGradientsOverOuterGradients()
	{
		// Uniform background with two gradient patches of equal magnitude:
		// Patch A (near center, dr = 1): horizontal gradient (direction 0°)
		// Patch B (far from center, dr = 8): vertical gradient (direction 90° -> 270°)
		using var image = new Image<L8>(50, 50);
		var centerRow = 25;
		var centerCol = 25;

		image.ProcessPixelRows(accessor =>
		{
			for (var r = 0; r < accessor.Height; r++)
			{
				var span = accessor.GetRowSpan(r);
				for (var c = 0; c < span.Length; c++)
					span[c] = new L8(128);
			}

			// Patch A at dr in [-1, 1], dc in [-2, 2]: horizontal gradient
			for (var r = centerRow - 1; r <= centerRow + 1; r++)
			{
				var span = accessor.GetRowSpan(r);
				for (var c = centerCol - 2; c <= centerCol + 2; c++)
					span[c] = new L8((byte)(128 + 20 * (c - centerCol)));
			}

			// Patch B at dr in [7, 8], dc in [-2, 2]: vertical gradient of same magnitude
			for (var r = centerRow + 7; r <= centerRow + 8; r++)
			{
				var span = accessor.GetRowSpan(r);
				for (var c = centerCol - 2; c <= centerCol + 2; c++)
					span[c] = new L8((byte)(128 + 20 * (r - (centerRow + 7))));
			}
		});

		var coordinate = new Coordinate(2, centerRow, centerCol);
		var keypoints = Sift.AssignOrientation(coordinate, image);

		Assert.NotEmpty(keypoints);
		// The primary peak must be determined by Patch A (~0°/360°) because of Gaussian distance decay e^(-r²/2σ²)
		var dominantKeypoint = keypoints[0];
		Assert.True(AngularDifference(dominantKeypoint.Orientation, 0.0) < 15.0,
			$"Dominant orientation should favor near-center gradient (~0°), but got {dominantKeypoint.Orientation}°");
	}
}