using PhotogrammetryLibrary;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Photogrammetry;

public static class Sift
{
	/// <summary>
	///     From an initial image, create a scale space over one octave (doubling of 𝛔).
	///     Each image after is blurred by a factor of K=2<sup>1/S</sup>
	/// </summary>
	/// <param name="initialImage">The base image, or bottom of the octave. Will be referenced in the returned list</param>
	/// <param name="s">Number of intervals in the octave, defaults to 3</param>
	/// <param name="sigma">Strength of initial blur applied, defaults to 1.6</param>
	/// <returns><paramref name="s" />+3 blurred images in the scale space of the image. </returns>
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
	///     Localizes and calculates the orientation of a keypoint
	/// </summary>
	/// <param name="extremaCoordinates">Location of potential keypoints, stored as (row, col, layer)</param>
	/// <param name="dogOctave">An octave of Difference of Gaussian Images (DoG)</param>
	/// <param name="gaussians"></param>
	/// <returns>List of stable keypoints</returns>
	public static List<Keypoint> LocalizeAndOrientKeypoints(List<int[]> extremaCoordinates, List<float[,]> dogOctave,
		List<Image<L8>> gaussians)
	{
		List<Keypoint> keypoints = [];

		foreach (var localizedCoordinate in extremaCoordinates
			         .Select(coordinate => LocalizeExtremum(dogOctave, coordinate)).OfType<int[]>())
			keypoints.AddRange(AssignOrientation(localizedCoordinate, gaussians[localizedCoordinate[2]]));

		return keypoints;
	}

	/// <summary>
	///     Localizes a potential keypoint by fitting to the nearby data in scale and location,
	///     then using that information to determine if a point should be rejected
	///     for low contrast or poor edge localization.
	/// </summary>
	/// <param name="dogOctave">An octave of Difference of Gaussian Images (DoG)</param>
	/// <param name="coordinate">Location of the potential keypoint stored as (row, col, layer)</param>
	/// <param name="r"></param>
	/// <returns>The localized keypoint stored as (row, col, layer), null if the keypoint isn't stable</returns>
	private static int[]? LocalizeExtremum(List<float[,]> dogOctave, int[] coordinate, int r = 10)
	{
		var coordinateWithOffset = (int[])coordinate.Clone();
		double[] gradient = null!;
		double[,] hessian = null!;
		double[]? offset = null!;
		var converged = false;

		for (var attempt = 0; attempt < 5; attempt++)
		{
			if (!Utils.IsWithinBounds(dogOctave, coordinateWithOffset)) return null;

			hessian = Utils.ApproximateHessian(dogOctave, coordinateWithOffset);
			gradient = Utils.ApproximateGradient(dogOctave, coordinateWithOffset);
			offset = Utils.CalculateOffset(hessian, gradient);

			if (offset == null) return null;

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

		if (!converged) return null;

		var row = coordinateWithOffset[0];
		var col = coordinateWithOffset[1];
		var layer = coordinateWithOffset[2];

		// remove low contrast points
		var contrast = dogOctave[layer][row, col] * Utils.ImageScale + 0.5 * (
			gradient[1] * offset[0] +
			gradient[0] * offset[1] +
			gradient[2] * offset[2]);

		if (Math.Abs(contrast) < 0.03) return null;

		// Eliminating edge responses
		var detH = hessian[0, 0] * hessian[1, 1] - Math.Pow(hessian[0, 1], 2);

		if (detH < 0) return null;

		var traceH = hessian[0, 0] + hessian[1, 1];

		return Math.Pow(traceH, 2) / detH > Math.Pow(r + 1, 2) / r ? null : coordinateWithOffset;
	}

	/// <summary>
	///     Calculates the orientation of a keypoint to achieve image rotation invariance
	/// </summary>
	/// <param name="coordinate">Location of the potential keypoint stored as (row, col, layer)</param>
	/// <param name="gaussianImage">The gaussian blurred image with the same scale as the keypoint</param>
	/// <returns></returns>
	public static List<Keypoint> AssignOrientation(int[] coordinate, Image<L8> gaussianImage)
	{
		var scale = 1.5 * coordinate[2];
		var radius = (int)Math.Round(3 * scale);
		const int bins = 36;
		var histogram = new double[bins];
		List<Keypoint> keypoints = [];

		for (var dr = -radius; dr <= radius; dr++)
		{
			var row = coordinate[0] + dr;

			if (row < 1 || row >= gaussianImage.Height - 1) continue;

			for (var dc = -radius; dc <= radius; dc++)
			{
				var col = coordinate[1] + dc;

				if (col < 1 || col >= gaussianImage.Width - 1) continue;

				var dx = gaussianImage[col + 1, row].PackedValue - gaussianImage[col - 1, row].PackedValue;
				var dy = gaussianImage[col, row + 1].PackedValue - gaussianImage[col, row - 1].PackedValue;
				var magnitude = Math.Sqrt(dx * dx + dy * dy);

				var orientation = Math.Atan2(dy, dx) * (180.0 / Math.PI);
				if (orientation < 0) orientation += 360.0;

				// gaussian weight centered on the keypoint
				var weight = Math.Exp(-(dr * dr + dc * dc) / (2.0 * Math.Pow(scale, 2))) /
				             (2.0 * Math.PI * Math.Pow(scale, 2));

				var histogramIndex = (int)Math.Round(orientation * bins / 360.0) % bins;
				if (histogramIndex < 0) histogramIndex += bins;
				histogram[histogramIndex] += weight * magnitude;
			}
		}

		var orientationMax = histogram.Max();
		if (orientationMax <= 0)
		{
			keypoints.Add(new Keypoint(coordinate[1], coordinate[0], coordinate[2], 0));
			return keypoints;
		}

		for (var i = 0; i < bins; i++)
		{
			var l = i == 0 ? bins - 1 : i - 1;
			var r = (i + 1) % bins;

			if (!(histogram[i] > histogram[l]) || !(histogram[i] > histogram[r]) ||
			    !(histogram[i] >= 0.8 * orientationMax)) continue;

			var peakBinIndex = 0.5 * (histogram[l] - histogram[r]) /
			                   (histogram[l] - 2.0 * histogram[i] + histogram[r]);
			peakBinIndex = i + peakBinIndex;
			peakBinIndex = peakBinIndex < 0 ? bins + peakBinIndex :
				peakBinIndex >= bins ? peakBinIndex - bins : peakBinIndex;

			var orientation = 360.0 - peakBinIndex * 360.0 / bins;

			keypoints.Add(new Keypoint(coordinate[1], coordinate[0], coordinate[2], orientation));
		}

		return keypoints;
	}
}