using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

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

	/// <summary>
	/// Creates a list of gaussian images matching the scale space dimensions.
	/// </summary>
	public static List<Image<L8>> CreateGaussians(int count = 6, int rows = 10, int cols = 10, byte fillValue = 128)
	{
		var list = new List<Image<L8>>(count);
		for (var i = 0; i < count; i++)
		{
			var img = new Image<L8>(cols, rows);
			if (fillValue != 0)
			{
				img.ProcessPixelRows(accessor =>
				{
					for (var y = 0; y < accessor.Height; y++)
					{
						var row = accessor.GetRowSpan(y);
						for (var x = 0; x < row.Length; x++)
							row[x] = new L8(fillValue);
					}
				});
			}
			list.Add(img);
		}
		return list;
	}
}

