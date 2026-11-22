using Xunit;

namespace Photogrammetry.Tests;

public class ExtremaDetectionTests
{
	[Fact]
	public void DetectLocalExtrema_IsolatedMaximum_DetectedSuccessfully()
	{
		var dogOctave = TestHelpers.CreateDogOctave(layers: 5, rows: 10, cols: 10, fillValue: 0f);
		dogOctave[2][5, 5] = 50f; // strictly greater than all 26 neighbors

		var extrema = Sift.DetectLocalExtrema(dogOctave);

		Assert.Single(extrema);
		Assert.Equal([5, 5, 2], extrema[0]); // [row, col, layer]
	}

	[Fact]
	public void DetectLocalExtrema_IsolatedMinimum_DetectedSuccessfully()
	{
		var dogOctave = TestHelpers.CreateDogOctave(layers: 5, rows: 10, cols: 10, fillValue: 0f);
		dogOctave[2][5, 5] = -50f; // strictly less than all 26 neighbors

		var extrema = Sift.DetectLocalExtrema(dogOctave);

		Assert.Single(extrema);
		Assert.Equal([5, 5, 2], extrema[0]); // [row, col, layer]
	}

	[Fact]
	public void DetectLocalExtrema_FlatUniformOctave_DetectsNothing()
	{
		var dogOctave = TestHelpers.CreateDogOctave(layers: 5, rows: 10, cols: 10, fillValue: 10f);

		var extrema = Sift.DetectLocalExtrema(dogOctave);

		Assert.Empty(extrema);
	}

	[Fact]
	public void DetectLocalExtrema_EqualNeighbor_PlateauNotDetected()
	{
		var dogOctave = TestHelpers.CreateDogOctave(layers: 5, rows: 10, cols: 10, fillValue: 0f);
		dogOctave[2][5, 5] = 50f;
		// One neighbor has the exact same value (plateau)
		dogOctave[2][5, 6] = 50f;

		var extrema = Sift.DetectLocalExtrema(dogOctave);

		Assert.Empty(extrema);
	}

	[Fact]
	public void DetectLocalExtrema_SaddlePoint_NotDetected()
	{
		var dogOctave = TestHelpers.CreateDogOctave(layers: 5, rows: 10, cols: 10, fillValue: 0f);
		dogOctave[2][5, 5] = 0f;
		dogOctave[2][5, 4] = -10f; // smaller neighbor
		dogOctave[2][5, 6] = 10f;  // larger neighbor

		var extrema = Sift.DetectLocalExtrema(dogOctave);

		// The saddle point at [5, 5, 2] has both smaller and larger neighbors, so it must not be detected
		Assert.DoesNotContain(extrema, c => c[0] == 5 && c[1] == 5 && c[2] == 2);
	}

	[Fact]
	public void DetectLocalExtrema_BoundaryPoints_NeverDetected()
	{
		var dogOctave = TestHelpers.CreateDogOctave(layers: 5, rows: 10, cols: 10, fillValue: 0f);

		// Peaks on borders
		dogOctave[0][5, 5] = 100f; // layer 0 (boundary scale)
		dogOctave[4][5, 5] = 100f; // layer 4 (boundary scale)
		dogOctave[2][0, 5] = 100f; // row 0 (top border)
		dogOctave[2][9, 5] = 100f; // row 9 (bottom border)
		dogOctave[2][5, 0] = 100f; // col 0 (left border)
		dogOctave[2][5, 9] = 100f; // col 9 (right border)

		var extrema = Sift.DetectLocalExtrema(dogOctave);

		Assert.Empty(extrema);
	}

	[Fact]
	public void DetectLocalExtrema_MultipleDistinctExtrema_FindsAll()
	{
		var dogOctave = TestHelpers.CreateDogOctave(layers: 5, rows: 20, cols: 20, fillValue: 0f);

		// Place two maxima and one minimum well separated from each other
		dogOctave[1][3, 3] = 40f;   // Max at [3, 3, 1]
		dogOctave[2][15, 15] = 80f; // Max at [15, 15, 2]
		dogOctave[3][8, 8] = -60f;  // Min at [8, 8, 3]

		var extrema = Sift.DetectLocalExtrema(dogOctave);

		Assert.Equal(3, extrema.Count);
		Assert.Contains(extrema, c => c[0] == 3 && c[1] == 3 && c[2] == 1);
		Assert.Contains(extrema, c => c[0] == 15 && c[1] == 15 && c[2] == 2);
		Assert.Contains(extrema, c => c[0] == 8 && c[1] == 8 && c[2] == 3);
	}
}
