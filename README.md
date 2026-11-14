Disk (huge images)
   ↓
Dataset (loads full image)
   ↓
Patch Sampler (smart, biased)
   ↓
Model
   ↓
Loss (focal / MSE on heatmap)
   ↓
Inference (sliding window + merge)
   ↓
detection
   ↓
