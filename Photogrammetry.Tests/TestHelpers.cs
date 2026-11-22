namespace Photogrammetry.Tests;

public static class TestHelpers
{
	/// <summary>
	/// Creates a list of 2D float arrays representing a Difference of Gaussians (DoG) octave.
	/// </summary>
	public static List<float[,]> CreateDogOctave(int layers = 5, int rows = 10, int cols = 10, float fillValue = 0f)
	{
		var octave = new List<float[,]>(layers);
		for (var l = 0; l < layers; l++)
		{
			var layer = new float[rows, cols];
			if (fillValue != 0f)
			{
				for (var r = 0; r < rows; r++)
				for (var c = 0; c < cols; c++)
					layer[r, c] = fillValue;
			}
			octave.Add(layer);
		}
		return octave;
	}
}
