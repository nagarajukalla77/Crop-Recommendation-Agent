from predictor import predict_crop

crop = predict_crop(
    90,
    42,
    43,
    20.8,
    82,
    6.5,
    202
)

print("Recommended Crop:", crop)