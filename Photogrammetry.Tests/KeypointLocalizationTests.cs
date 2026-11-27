using PhotogrammetryLibrary;
using Xunit;

namespace Photogrammetry.Tests;

public class KeypointLocalizationTests
{
	[Fact]
	public void LocalizeAndOrientKeypoints_LowContrastExtremum_IsRejected()
	{
		var dogOctave = TestHelpers.CreateDogOctave(layers: 5, rows: 10, cols: 10, fillValue: 0f);
		var gaussians = TestHelpers.CreateGaussians(count: 6, rows: 10, cols: 10);

		// A very weak peak: center is 1.0f (contrast ~ 1/255 = 0.0039, well below 0.03 threshold)
		dogOctave[2][5, 5] = 1.0f;
		var extrema = new List<int[]> { new[] { 5, 5, 2 } };

		var keypoints = Sift.LocalizeAndOrientKeypoints(extrema, dogOctave, gaussians);

		Assert.Empty(keypoints);
	}

	[Fact]
	public void LocalizeAndOrientKeypoints_HighContrastBlob_IsRetained()
	{
		var dogOctave = TestHelpers.CreateDogOctave(layers: 5, rows: 10, cols: 10, fillValue: 0f);
		var gaussians = TestHelpers.CreateGaussians(count: 6, rows: 10, cols: 10);

		// Create a strong, symmetric paraboloid peak centered at [5, 5, 2]:
		// Peak value 60f (contrast ~ 60/255 = 0.235 > 0.03)
		for (var l = 0; l < 5; l++)
		for (var r = 0; r < 10; r++)
		for (var c = 0; c < 10; c++)
		{
			var dr = r - 5;
			var dc = c - 5;
			var dl = l - 2;
			var val = 60f - 6f * (dr * dr + dc * dc + dl * dl);
			dogOctave[l][r, c] = Math.Max(0f, val);
		}

		var extrema = new List<int[]> { new[] { 5, 5, 2 } };

		var keypoints = Sift.LocalizeAndOrientKeypoints(extrema, dogOctave, gaussians);

		Assert.NotEmpty(keypoints);
		Assert.All(keypoints, kp =>
		{
			Assert.Equal(5, kp.X);
			Assert.Equal(5, kp.Y);
			Assert.Equal(2, kp.S);
		});
	}

	[Fact]
	public void LocalizeAndOrientKeypoints_SingularHessian_IsRejected()
	{
		var dogOctave = TestHelpers.CreateDogOctave(layers: 5, rows: 10, cols: 10, fillValue: 0f);
		var gaussians = TestHelpers.CreateGaussians(count: 6, rows: 10, cols: 10);

		// A linear ramp has zero second derivatives, causing Hessian to be all zeros (singular)
		for (var l = 0; l < 5; l++)
		for (var r = 0; r < 10; r++)
		for (var c = 0; c < 10; c++)
			dogOctave[l][r, c] = 10f * r + 10f * c;

		var extrema = new List<int[]> { new[] { 5, 5, 2 } };

		var keypoints = Sift.LocalizeAndOrientKeypoints(extrema, dogOctave, gaussians);

		// With singular Hessian, CalculateOffset returns null, rejecting the keypoint
		Assert.Empty(keypoints);
	}

	[Fact]
	public void LocalizeAndOrientKeypoints_NegativeDeterminant_IsRejected()
	{
		var dogOctave = TestHelpers.CreateDogOctave(layers: 5, rows: 10, cols: 10, fillValue: 0f);
		var gaussians = TestHelpers.CreateGaussians(count: 6, rows: 10, cols: 10);

		// Saddle-like curvature around center: dxx > 0, dyy < 0 => detH < 0
		// f(r, c, l) = 50 + 10*(c-5)^2 - 10*(r-5)^2 - 5*(l-2)^2
		for (var l = 0; l < 5; l++)
		for (var r = 0; r < 10; r++)
		for (var c = 0; c < 10; c++)
		{
			var dr = r - 5;
			var dc = c - 5;
			var dl = l - 2;
			dogOctave[l][r, c] = 50f + 10f * (dc * dc) - 10f * (dr * dr) - 5f * (dl * dl);
		}

		var extrema = new List<int[]> { new[] { 5, 5, 2 } };

		var keypoints = Sift.LocalizeAndOrientKeypoints(extrema, dogOctave, gaussians);

		Assert.Empty(keypoints);
	}

	[Fact]
	public void LocalizeAndOrientKeypoints_OutOfBoundsShift_IsRejected()
	{
		var dogOctave = TestHelpers.CreateDogOctave(layers: 5, rows: 10, cols: 10, fillValue: 0f);
		var gaussians = TestHelpers.CreateGaussians(count: 6, rows: 10, cols: 10);

		// Paraboloid centered at row -2 (outside valid bounds):
		// This creates a strong directional gradient pulling the coordinate past the row=0 boundary
		for (var l = 0; l < 5; l++)
		for (var r = 0; r < 10; r++)
		for (var c = 0; c < 10; c++)
		{
			var dr = r - (-2);
			var dc = c - 5;
			var dl = l - 2;
			var val = 100f - 5f * (dr * dr + dc * dc + dl * dl);
			dogOctave[l][r, c] = Math.Max(0f, val);
		}

		// Initial coordinate at valid interior point [2, 5, 2]
		var extrema = new List<int[]> { new[] { 2, 5, 2 } };

		var keypoints = Sift.LocalizeAndOrientKeypoints(extrema, dogOctave, gaussians);

		// Coordinate shifts towards negative rows and is discarded when going out of bounds
		Assert.Empty(keypoints);
	}

	[Fact]
	public void LocalizeAndOrientKeypoints_ShiftsCoordinateToPeakOnConvergence()
	{
		var dogOctave = TestHelpers.CreateDogOctave(layers: 5, rows: 15, cols: 15, fillValue: 0f);
		var gaussians = TestHelpers.CreateGaussians(count: 6, rows: 15, cols: 15);

		// Peak is at [6, 6, 2], but initial guess is at [5, 5, 2].
		// Paraboloid centered at (6, 6, 2):
		for (var l = 0; l < 5; l++)
		for (var r = 0; r < 15; r++)
		for (var c = 0; c < 15; c++)
		{
			var dr = r - 6;
			var dc = c - 6;
			var dl = l - 2;
			var val = 80f - 5f * (dr * dr + dc * dc + dl * dl);
			dogOctave[l][r, c] = Math.Max(0f, val);
		}

		var initialCoordinate = new[] { 5, 5, 2 };
		var extrema = new List<int[]> { initialCoordinate };

		var keypoints = Sift.LocalizeAndOrientKeypoints(extrema, dogOctave, gaussians);

		Assert.NotEmpty(keypoints);
		// LocalizeExtremum should iterate Newton steps and shift coordinate to the true peak at (col 6, row 6, layer 2)
		Assert.All(keypoints, kp =>
		{
			Assert.Equal(6, kp.X);
			Assert.Equal(6, kp.Y);
			Assert.Equal(2, kp.S);
		});
	}
}

