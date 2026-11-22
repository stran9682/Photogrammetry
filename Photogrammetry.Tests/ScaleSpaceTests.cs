using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Xunit;

namespace Photogrammetry.Tests;

public class ScaleSpaceTests
{
	[Theory]
	[InlineData(2, 5)]
	[InlineData(3, 6)]
	[InlineData(4, 7)]
	public void CreateScaleSpace_ReturnsExpectedNumberOfOctaveLayers(int s, int expectedCount)
	{
		using var image = new Image<L8>(32, 32);
		var scaleSpace = Sift.CreateScaleSpace(image, s: s);

		Assert.Equal(expectedCount, scaleSpace.Count);
	}

	[Fact]
	public void CreateScaleSpace_FirstImageIsOriginalInstance()
	{
		using var image = new Image<L8>(32, 32);
		var scaleSpace = Sift.CreateScaleSpace(image);

		Assert.Same(image, scaleSpace[0]);
	}

	[Fact]
	public void CreateScaleSpace_PreservesImageDimensionsAcrossLayers()
	{
		const int width = 48;
		const int height = 36;
		using var image = new Image<L8>(width, height);
		var scaleSpace = Sift.CreateScaleSpace(image);

		foreach (var layer in scaleSpace)
		{
			Assert.Equal(width, layer.Width);
			Assert.Equal(height, layer.Height);
		}
	}

	[Fact]
	public void CreateScaleSpace_ProgressivelyBlursImpulseFeature()
	{
		// An impulse in the center should spread outwards with each successive Gaussian blur,
		// causing the peak center intensity to monotonically decrease across layers.
		const int size = 33;
		using var image = new Image<L8>(size, size);
		var center = size / 2;
		image[center, center] = new L8(255);

		var scaleSpace = Sift.CreateScaleSpace(image, s: 3);

		byte prevPeak = 255;
		for (var i = 1; i < scaleSpace.Count; i++)
		{
			var currentPeak = scaleSpace[i][center, center].PackedValue;
			Assert.True(currentPeak < prevPeak,
				$"Expected peak at scale {i} ({currentPeak}) to be less than scale {i - 1} ({prevPeak}).");
			prevPeak = currentPeak;
		}
	}

	[Fact]
	public void CreateDoGOctave_ReturnsFiveDifferenceLayersForDefaultScaleSpace()
	{
		using var image = new Image<L8>(40, 30);
		var scaleSpace = Sift.CreateScaleSpace(image, s: 3);

		var dogOctave = Sift.CreateDoGOctave(scaleSpace);

		Assert.Equal(5, dogOctave.Count);
		foreach (var dog in dogOctave)
		{
			Assert.Equal(30, dog.GetLength(0)); // Height / rows
			Assert.Equal(40, dog.GetLength(1)); // Width / cols
		}
	}

	[Fact]
	public void CreateDoGOctave_IdenticalImages_ProducesAllZeroDifferences()
	{
		// When scale space consists of identical images, DoG differences must all be zero
		var scaleSpace = new List<Image<L8>>();
		for (var i = 0; i < 6; i++)
		{
			var img = new Image<L8>(10, 10);
			img.ProcessPixelRows(accessor =>
			{
				for (var r = 0; r < accessor.Height; r++)
				{
					var row = accessor.GetRowSpan(r);
					for (var c = 0; c < row.Length; c++)
						row[c] = new L8(128);
				}
			});
			scaleSpace.Add(img);
		}

		var dogOctave = Sift.CreateDoGOctave(scaleSpace);

		Assert.Equal(5, dogOctave.Count);
		foreach (var dog in dogOctave)
		{
			for (var r = 0; r < 10; r++)
			for (var c = 0; c < 10; c++)
				Assert.Equal(0f, dog[r, c]);
		}
	}

	[Fact]
	public void CreateDoGOctave_ComputesAccuratePixelDifferences()
	{
		var scaleSpace = new List<Image<L8>>();
		for (var i = 0; i < 6; i++)
		{
			var img = new Image<L8>(8, 8);
			byte val = (byte)(50 + i * 20); // 50, 70, 90, 110, 130, 150
			img.ProcessPixelRows(accessor =>
			{
				for (var r = 0; r < accessor.Height; r++)
				{
					var row = accessor.GetRowSpan(r);
					for (var c = 0; c < row.Length; c++)
						row[c] = new L8(val);
				}
			});
			scaleSpace.Add(img);
		}

		var dogOctave = Sift.CreateDoGOctave(scaleSpace);

		// Difference between consecutive images should be +20 for each layer
		Assert.Equal(5, dogOctave.Count);
		foreach (var dog in dogOctave)
		{
			for (var r = 0; r < 8; r++)
			for (var c = 0; c < 8; c++)
				Assert.Equal(20f, dog[r, c]);
		}
	}
}
