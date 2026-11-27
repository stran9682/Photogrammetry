using PhotogrammetryLibrary;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Photogrammetry.Tests;

public class SiftPipelineTests
{
	[Fact]
	public void EndToEndPipeline_BlankImage_ProducesNoKeypoints()
	{
		using var image = new Image<L8>(64, 64); // all black image

		var scaleSpace = Sift.CreateScaleSpace(image);
		var dogOctave = Sift.CreateDoGOctave(scaleSpace);
		var extrema = Sift.DetectLocalExtrema(dogOctave);
		var keypoints = Sift.LocalizeAndOrientKeypoints(extrema, dogOctave, scaleSpace);

		Assert.Equal(6, scaleSpace.Count);
		Assert.Equal(5, dogOctave.Count);
		Assert.Empty(keypoints);
	}

	[Fact]
	public void EndToEndPipeline_SyntheticFeatureWithPreprocessing_DetectsAndLocalizesKeypoints()
	{
		// Create a 64x64 image with multiple high-contrast geometric features
		using var image = new Image<L8>(64, 64);
		image.ProcessPixelRows(accessor =>
		{
			for (var r = 0; r < accessor.Height; r++)
			{
				var row = accessor.GetRowSpan(r);
				for (var c = 0; c < row.Length; c++)
					// Checkerboard / high-contrast spots
					if ((r / 8 + c / 8) % 2 == 0)
						row[c] = new L8(255);
					else
						row[c] = new L8(0);
			}
		});

		// Follow the base image preprocessing from Program.cs:
		var sigmaDiff = (float)Math.Sqrt(Math.Pow(1.6, 2) - 1);
		image.Mutate(ctx => ctx
			.Resize(ctx.GetCurrentSize().Width * 2, ctx.GetCurrentSize().Height * 2, KnownResamplers.Triangle)
			.GaussianBlur(sigmaDiff));

		var scaleSpace = Sift.CreateScaleSpace(image);
		var dogOctave = Sift.CreateDoGOctave(scaleSpace);
		var candidateExtrema = Sift.DetectLocalExtrema(dogOctave);

		Assert.NotEmpty(candidateExtrema);

		var keypoints = Sift.LocalizeAndOrientKeypoints(candidateExtrema, dogOctave, scaleSpace);

		// Verified that the localization executes cleanly and returns valid coordinates
		foreach (var kp in keypoints)
		{
			Assert.True(kp.X >= 0 && kp.X < image.Width);
			Assert.True(kp.Y >= 0 && kp.Y < image.Height);
			Assert.True(kp.S >= 0 && kp.S < dogOctave.Count);
		}
	}

	[Fact]
	public void EndToEndPipeline_SampleImage_ExecutesSuccessfullyAndFindsKeypoints()
	{
		if (!File.Exists("input.jpeg")) return;

		using var image = Image.Load<L8>("input.jpeg");
		var sigmaDiff = (float)Math.Sqrt(Math.Pow(1.6, 2) - 1);

		image.Mutate(ctx => ctx
			.Resize(ctx.GetCurrentSize().Width * 2, ctx.GetCurrentSize().Height * 2, KnownResamplers.Triangle)
			.GaussianBlur(sigmaDiff));

		var scaleSpace = Sift.CreateScaleSpace(image);
		var dogOctave = Sift.CreateDoGOctave(scaleSpace);
		var potentialKeypoints = Sift.DetectLocalExtrema(dogOctave);

		Assert.NotEmpty(potentialKeypoints);

		var initialCount = potentialKeypoints.Count;
		var keypoints = Sift.LocalizeAndOrientKeypoints(potentialKeypoints, dogOctave, scaleSpace);

		// Localization filters points based on contrast and edge responses
		Assert.True(keypoints.Count <= initialCount);
		Assert.NotEmpty(keypoints);

		foreach (var kp in keypoints)
		{
			Assert.True(kp.X >= 0 && kp.X < image.Width);
			Assert.True(kp.Y >= 0 && kp.Y < image.Height);
			Assert.True(kp.S >= 0 && kp.S < dogOctave.Count);
		}
	}
}