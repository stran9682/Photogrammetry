using PhotogrammetryLibrary;

namespace Photogrammetry;

public static class Utils
{
	/// <summary>
	///     The SIFT paper assumes pixel values in range [0,1].
	///     To bring L8 images to scale matching the paper, multiply by 1/255.
	/// </summary>
	public const double ImageScale = 1.0 / 255.0;

	/// <summary>
	///     Uses a second order taylor series to approximate the gradient at a point.
	/// </summary>
	/// <param name="dogOctave">An octave of Difference of Gaussian Images (DoG)</param>
	/// <param name="coordinate">Location of the potential keypoint stored as (row, col, layer)</param>
	/// <returns>The gradient as [dx, dy, ds]</returns>
	/// <remarks>The order of the gradient is different from the coordinate.</remarks>
	public static double[] ApproximateGradient(List<float[,]> dogOctave, Coordinate coordinate)
	{
		var row = coordinate.Row;
		var col = coordinate.Column;
		var layer = coordinate.Layer;

		var dx = ImageScale * 0.5 * (dogOctave[layer][row, col + 1] - dogOctave[layer][row, col - 1]);
		var dy = ImageScale * 0.5 * (dogOctave[layer][row + 1, col] - dogOctave[layer][row - 1, col]);
		var ds = ImageScale * 0.5 * (dogOctave[layer + 1][row, col] - dogOctave[layer - 1][row, col]);

		return [dx, dy, ds];
	}

	public static double[,] ApproximateHessian(List<float[,]> dogOctave, Coordinate coordinate)
	{
		var row = coordinate.Row;
		var col = coordinate.Column;
		var layer = coordinate.Layer;

		var center = dogOctave[layer][row, col];
		var dxx = ImageScale * (dogOctave[layer][row, col + 1] - 2 * center + dogOctave[layer][row, col - 1]);
		var dyy = ImageScale * (dogOctave[layer][row + 1, col] - 2 * center + dogOctave[layer][row - 1, col]);
		var dss = ImageScale * (dogOctave[layer + 1][row, col] - 2 * center + dogOctave[layer - 1][row, col]);

		var dxy = ImageScale * 0.25 *
		          (dogOctave[layer][row + 1, col + 1] - dogOctave[layer][row + 1, col - 1] -
			          dogOctave[layer][row - 1, col + 1] + dogOctave[layer][row - 1, col - 1]);
		var dxs = ImageScale * 0.25 *
		          (dogOctave[layer + 1][row, col + 1] - dogOctave[layer + 1][row, col - 1] -
			          dogOctave[layer - 1][row, col + 1] + dogOctave[layer - 1][row, col - 1]);
		var dys = ImageScale * 0.25 *
		          (dogOctave[layer + 1][row + 1, col] - dogOctave[layer + 1][row - 1, col] -
			          dogOctave[layer - 1][row + 1, col] + dogOctave[layer - 1][row - 1, col]);

		double[,] hessian =
		{
			{ dxx, dxy, dxs },
			{ dxy, dyy, dys },
			{ dxs, dys, dss }
		};

		return hessian;
	}

	/// <summary>
	///     Calculates the offset of a sample point from the estimated location of the extremum using Cramer's Method.
	///     Solves equation 3 in the SIFT paper.
	/// </summary>
	/// <param name="hessian">The approximated hessian at the point</param>
	/// <param name="gradient">The approximated gradient at the point</param>
	/// <returns> Returns [rowOffset, colOffset, scaleOffset] </returns>
	public static double[]? CalculateOffset(double[,] hessian, double[] gradient)
	{
		var hessianDet =
			hessian[0, 0] * (hessian[1, 1] * hessian[2, 2] - hessian[1, 2] * hessian[2, 1]) -
			hessian[0, 1] * (hessian[1, 0] * hessian[2, 2] - hessian[1, 2] * hessian[2, 0]) +
			hessian[0, 2] * (hessian[1, 0] * hessian[2, 1] - hessian[1, 1] * hessian[2, 0]);

		if (Math.Abs(hessianDet) < 1e-10) return null; // matrix is singular

		var xDet =
			gradient[0] * (hessian[1, 1] * hessian[2, 2] - hessian[1, 2] * hessian[2, 1]) -
			hessian[0, 1] * (gradient[1] * hessian[2, 2] - hessian[1, 2] * gradient[2]) +
			hessian[0, 2] * (gradient[1] * hessian[2, 1] - hessian[1, 1] * gradient[2]);

		var yDet =
			hessian[0, 0] * (gradient[1] * hessian[2, 2] - hessian[1, 2] * gradient[2]) -
			gradient[0] * (hessian[1, 0] * hessian[2, 2] - hessian[1, 2] * hessian[2, 0]) +
			hessian[0, 2] * (hessian[1, 0] * gradient[2] - gradient[1] * hessian[2, 0]);

		var sDet =
			hessian[0, 0] * (hessian[1, 1] * gradient[2] - gradient[1] * hessian[2, 1]) -
			hessian[0, 1] * (hessian[1, 0] * gradient[2] - gradient[1] * hessian[2, 0]) +
			gradient[0] * (hessian[1, 0] * hessian[2, 1] - hessian[1, 1] * hessian[2, 0]);

		// SIFT Taylor series extremum is x̂ = -H⁻¹g
		return [-yDet / hessianDet, -xDet / hessianDet, -sDet / hessianDet];
	}

	public static bool IsWithinBounds(List<float[,]> dogOctave, Coordinate coordinate)
	{
		var row = coordinate.Row;
		var col = coordinate.Column;
		var layer = coordinate.Layer;

		if (layer < 1 || layer >= dogOctave.Count - 1) return false;

		var rowLen = dogOctave[layer].GetLength(0);
		if (row < 1 || row >= rowLen - 1) return false;

		var colLen = dogOctave[layer].GetLength(1);
		return col >= 1 && col < colLen - 1;
	}
}