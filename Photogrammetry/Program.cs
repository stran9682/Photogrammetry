using Photogrammetry;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

var image = Image.Load<L8>("input.jpeg"); // storing pixels as L8 converts to greyscale

// Produce base image by doubling the size of the input image
// this creates keypoints which would've otherwise been lost
// starting with the original image then immediately applying a blur
var sigmaDiff = (float)Math.Sqrt(Math.Pow(1.6, 2) - 1);

image.Mutate(ctx => ctx
	.Resize(ctx.GetCurrentSize().Width * 2, ctx.GetCurrentSize().Height * 2, KnownResamplers.Triangle)
	.GaussianBlur(sigmaDiff));

// Continue blurring by factor of k = 2^(1/s), where s = 3.
// We'll produce s+3 total blurred images (6).
var scaleSpace = Sift.CreateScaleSpace(image);

// produce the difference of gaussians for the octave
var diffOfGaussians = Sift.CreateDoGOctave(scaleSpace);

// search for local scale space extrema.
// Potential features will either be smaller or larger than their all surrounding neighbors
var potentialKeypoints = Sift.DetectLocalExtrema(diffOfGaussians);

Sift.LocalizeKeypoints(diffOfGaussians, potentialKeypoints);