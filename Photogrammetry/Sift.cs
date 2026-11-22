using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Photogrammetry;

public static class Sift
{
	/// <summary>
	///     From an initial image, create a scale space over one octave.
	///     Each image after is blurred by a factor of K=2<sup>1/S</sup>
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
		for (var i = 1; i < scaleSpace.Count; i++)
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
						dog[row, col] = currentRow[col].PackedValue - prevRow[col].PackedValue;
				}
			});

			diffOfGaussians.Add(dog);
		}

		return diffOfGaussians;
	}

	/// <summary>
	///     Search for local scale space extrema.
	///     Potential features will either be smaller or larger than all 26 surrounding neighbors.
	/// </summary>
	/// <param name="dogOctave">An octave of Difference of Gaussian Images (DoG)</param>
	/// <returns>Locations of potential keypoints, stored as [row, col, layer]</returns>
	public static List<int[]> DetectLocalExtrema(List<float[,]> dogOctave)
	{
		var result = new List<int[]>();

		for (var layer = 1; layer < dogOctave.Count - 1; layer++)
		{
			var currentImage = dogOctave[layer];

			for (var row = 1; row < currentImage.GetLength(0) - 1; row++)
			for (var col = 1; col < currentImage.GetLength(1) - 1; col++)
				if (IsExtrema(dogOctave, row, col, layer))
					result.Add([row, col, layer]);
		}

		return result;
	}

	private static bool IsExtrema(List<float[,]> dogOctave, int row, int col, int layer)
	{
		var center = dogOctave[layer][row, col];
		bool isMax = true, isMin = true;

		for (var dl = -1; dl <= 1; dl++)
		{
			var image = dogOctave[layer + dl];

			for (var dr = -1; dr <= 1; dr++)
			for (var dc = -1; dc <= 1; dc++)
			{
				if (dl == 0 && dr == 0 && dc == 0) continue;

				var val = image[row + dr, col + dc];
				if (val >= center) isMax = false;
				if (val <= center) isMin = false;

				if (!isMax && !isMin) return false;
			}
		}

		return isMax || isMin;
	}

	/// <summary>
	///     Localizes potential keypoints by fitting to the nearby data in scale and location.
	///     Then using that information,
	///     determines if a point should be removed for low contrast or poor edge localization.
	/// </summary>
	/// <param name="dogOctave">An octave of Difference of Gaussian Images (DoG)</param>
	/// <param name="extremaCoordinates">Location of potential keypoints, stored as (row, col, layer)</param>
	/// <remarks>Will modify extremaCoordinates in place</remarks>
	public static void LocalizeKeypoints(List<float[,]> dogOctave, List<int[]> extremaCoordinates)
	{
		// will also localize the extrema
		extremaCoordinates.RemoveAll(c => !LocalizeExtremum(dogOctave, c));
	}

	/// <summary>
	///     Localizes a potential keypoint by fitting to the nearby data in scale and location.
	///     Then using that information,
	///     determines if a point should be rejected for low contrast or poor edge localization.
	/// </summary>
	/// <param name="dogOctave">An octave of Difference of Gaussian Images (DoG)</param>
	/// <param name="coordinate">Location of the potential keypoint stored as (row, col, layer)</param>
	/// <param name="r"></param>
	/// <returns>true if this keypoint should be kept, false otherwise.</returns>
	private static bool LocalizeExtremum(List<float[,]> dogOctave, int[] coordinate, int r = 10)
	{
		var coordinateWithOffset = (int[])coordinate.Clone();
		double[] gradient = null!;
		double[,] hessian = null!;
		double[]? offset = null!;
		var converged = false;

		for (var attempt = 0; attempt < 5; attempt++)
		{
			if (!Utils.IsWithinBounds(dogOctave, coordinateWithOffset)) return false;

			hessian = Utils.ApproximateHessian(dogOctave, coordinateWithOffset);
			gradient = Utils.ApproximateGradient(dogOctave, coordinateWithOffset);
			offset = Utils.CalculateOffset(hessian, gradient);

			if (offset == null) return false;

			// If offset is <= 0.5 in all dimensions, extremum lies within this sample pixel cube
			if (Math.Abs(offset[0]) <= 0.5 && Math.Abs(offset[1]) <= 0.5 && Math.Abs(offset[2]) <= 0.5)
			{
				converged = true;
				break;
			}

			coordinateWithOffset[0] += (int)Math.Round(offset[0]);
			coordinateWithOffset[1] += (int)Math.Round(offset[1]);
			coordinateWithOffset[2] += (int)Math.Round(offset[2]);
		}

		if (!converged) return false;

		coordinate[0] = coordinateWithOffset[0];
		coordinate[1] = coordinateWithOffset[1];
		coordinate[2] = coordinateWithOffset[2];

		var row = coordinateWithOffset[0];
		var col = coordinateWithOffset[1];
		var layer = coordinateWithOffset[2];

		// remove low contrast points
		var contrast = dogOctave[layer][row, col] * Utils.ImageScale + 0.5 * (
			gradient[1] * offset[0] +
			gradient[0] * offset[1] +
			gradient[2] * offset[2]);

		if (Math.Abs(contrast) < 0.03) return false;

		// Eliminating edge responses
		var detH = hessian[0, 0] * hessian[1, 1] - Math.Pow(hessian[0, 1], 2);

		if (detH < 0) return false;

		var traceH = hessian[0, 0] + hessian[1, 1];

		return Math.Pow(traceH, 2) / detH < Math.Pow(r + 1, 2) / r;
	}
}