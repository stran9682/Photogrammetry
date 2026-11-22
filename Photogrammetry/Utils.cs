namespace Photogrammetry;

public static class Utils
{
	public const double ImageScale = 1.0 / 255.0;

	public static double[] ApproximateGradient(List<float[,]> dogOctave, int[] coordinate)
	{
		var row = coordinate[0];
		var col = coordinate[1];
		var layer = coordinate[2];

		var dx = ImageScale * 0.5 * (dogOctave[layer][row, col + 1] - dogOctave[layer][row, col - 1]);
		var dy = ImageScale * 0.5 * (dogOctave[layer][row + 1, col] - dogOctave[layer][row - 1, col]);
		var ds = ImageScale * 0.5 * (dogOctave[layer + 1][row, col] - dogOctave[layer - 1][row, col]);

		return [dx, dy, ds];
	}

	public static double[,] ApproximateHessian(List<float[,]> dogOctave, int[] coordinate)
	{
		var row = coordinate[0];
		var col = coordinate[1];
		var layer = coordinate[2];

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
		// Returns [rowOffset, colOffset, scaleOffset] to match [row, col, layer] coordinate ordering
		return [-yDet / hessianDet, -xDet / hessianDet, -sDet / hessianDet];
	}

	public static bool IsWithinBounds(List<float[,]> dogOctave, int[] coordinate)
	{
		var row = coordinate[0];
		var col = coordinate[1];
		var layer = coordinate[2];

		if (layer < 1 || layer >= dogOctave.Count - 1) return false;

		var rowLen = dogOctave[layer].GetLength(0);
		if (row < 1 || row >= rowLen - 1) return false;

		var colLen = dogOctave[layer].GetLength(1);
		return col >= 1 && col < colLen - 1;
	}
}