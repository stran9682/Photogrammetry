using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Photogrammetry;

public static class Sift
{
	/// <summary>
	/// From an initial image, create a scale space over one octave.
	/// Each image after is blurred by a factor of K=2<sup>1/S</sup>
	/// </summary>
	/// <param name="initialImage">The base image, or bottom of the octave. Will be referenced in the returned list</param>
	/// <param name="s">Number of searchable images in the octave, defaults to 3</param>
	/// <param name="sigma">Strength of initial blur applied, defaults to 1.6</param>
	/// <returns>Scale space of the image, indexed by power of K</returns>
	public static List<Image<L8>> CreateScaleSpace(Image<L8> initialImage, int s = 3, double sigma = 1.6)
	{
		List<Image<L8>> scaleSpace = [initialImage];
		var k = Math.Pow(2, 1.0 / s);

		for (var i = 1; i < s + 3; i++)
		{
			var prevSigma = Math.Pow(k, i - 1) * sigma;
			var nextSigma = k * prevSigma;
			var diff = (float)Math.Sqrt(Math.Pow(nextSigma, 2) - Math.Pow(prevSigma, 2));

			var prevImage = scaleSpace[i - 1];
			scaleSpace.Add(prevImage.Clone(ctx => ctx.GaussianBlur(diff)));
		}

		return scaleSpace;
	}

	public static List<float[,]> CreateDoGOctave(List<Image<L8>> scaleSpace)
	{
		var diffOfGaussians = new List<float[,]>();
		for (var i = 1; i < 6; i++)
		{
			var prevImage = scaleSpace[i - 1];
			var currentImage = scaleSpace[i];
			var dog = new float[currentImage.Height, currentImage.Width];

			currentImage.ProcessPixelRows(prevImage, (currentAccessor, prevAccessor) =>
			{
				for (var row = 0; row < prevAccessor.Height; row++)
				{
					var prevRow = prevAccessor.GetRowSpan(row);
					var currentRow = currentAccessor.GetRowSpan(row);

					for (var col = 0; col < prevRow.Length; col++)
					{
						dog[row, col] = currentRow[col].PackedValue - prevRow[col].PackedValue;
					}
				}
			});

			diffOfGaussians.Add(dog);
		}

		return diffOfGaussians;
	}

	/// <summary>
	/// Search for local scale space extrema.
	/// Potential features will be smaller or larger than all their surrounding neighbors.
	/// </summary>
	/// <param name="dogOctave">An octave of Difference of Gaussian Images (DoG)</param>
	public static void DetectLocalExtrema(List<float[,]> dogOctave)
	{
		for (var i = 1; i < dogOctave.Count - 1; i++)
		{
			var currentImage = dogOctave[i];

			for (var row = 1; row < currentImage.GetLength(0) - 1; row++)
			{
				for (var col = 1; col < currentImage.GetLength(1) - 1; col++)
				{
					if (IsExtrema(dogOctave, i, row, col))
					{
						// TODO: record the local extrema
					}
				}
			}
		}
	}

	private static bool IsExtrema(List<float[,]> dogOctave, int layer, int row, int col)
	{
		var center = dogOctave[layer][row, col];
		bool isMax = true, isMin = true;

		for (var dl = -1; dl <= 1; dl++)
		{
			var image = dogOctave[layer + dl];

			for (var dr = -1; dr <= 1; dr++)
			{
				for (var dc = -1; dc <= 1; dc++)
				{
					if (dl == 0 && dr == 0 && dc == 0) continue;

					var val = image[row + dr, col + dc];
					if (val >= center) isMax = false;
					if (val <= center) isMin = false;

					if (!isMax && !isMin) return false;
				}
			}
		}

		return isMax || isMin;
	}
}