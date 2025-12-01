import re
import pandas as pd
import matplotlib.pyplot as plt

log_text = """
TCN Epoch   1 | Train Loss 0.8081 | Val AUC 0.5151 | Val Acc 0.5072
TCN Epoch   2 | Train Loss 0.7187 | Val AUC 0.5186 | Val Acc 0.5117
TCN Epoch   3 | Train Loss 0.7038 | Val AUC 0.5245 | Val Acc 0.5133
TCN Epoch   4 | Train Loss 0.6964 | Val AUC 0.5275 | Val Acc 0.5174
TCN Epoch   5 | Train Loss 0.6939 | Val AUC 0.5187 | Val Acc 0.5110
TCN Epoch   6 | Train Loss 0.6931 | Val AUC 0.5251 | Val Acc 0.5179
TCN Epoch   7 | Train Loss 0.6906 | Val AUC 0.5253 | Val Acc 0.5177
TCN Epoch   8 | Train Loss 0.6880 | Val AUC 0.5283 | Val Acc 0.5180
TCN Epoch   9 | Train Loss 0.6871 | Val AUC 0.5327 | Val Acc 0.5217
TCN Epoch  10 | Train Loss 0.6857 | Val AUC 0.5287 | Val Acc 0.5190
TCN Epoch  11 | Train Loss 0.6855 | Val AUC 0.5326 | Val Acc 0.5201
TCN Epoch  12 | Train Loss 0.6846 | Val AUC 0.5268 | Val Acc 0.5167
TCN Epoch  13 | Train Loss 0.6841 | Val AUC 0.5260 | Val Acc 0.5183
TCN Epoch  14 | Train Loss 0.6819 | Val AUC 0.5288 | Val Acc 0.5206
TCN Epoch  15 | Train Loss 0.6808 | Val AUC 0.5192 | Val Acc 0.5157
TCN Epoch  16 | Train Loss 0.6803 | Val AUC 0.5275 | Val Acc 0.5181
TCN Epoch  17 | Train Loss 0.6795 | Val AUC 0.5300 | Val Acc 0.5212
TCN Epoch  18 | Train Loss 0.6773 | Val AUC 0.5318 | Val Acc 0.5236
TCN Epoch  19 | Train Loss 0.6777 | Val AUC 0.5273 | Val Acc 0.5186
TCN Epoch  20 | Train Loss 0.6764 | Val AUC 0.5279 | Val Acc 0.5185
TCN Epoch  21 | Train Loss 0.6741 | Val AUC 0.5276 | Val Acc 0.5208
TCN Epoch  22 | Train Loss 0.6723 | Val AUC 0.5231 | Val Acc 0.5166
TCN Epoch  23 | Train Loss 0.6731 | Val AUC 0.5276 | Val Acc 0.5198
TCN Epoch  24 | Train Loss 0.6710 | Val AUC 0.5272 | Val Acc 0.5192
TCN Epoch  25 | Train Loss 0.6683 | Val AUC 0.5264 | Val Acc 0.5220
TCN Epoch  26 | Train Loss 0.6685 | Val AUC 0.5247 | Val Acc 0.5191
TCN Epoch  27 | Train Loss 0.6655 | Val AUC 0.5304 | Val Acc 0.5225
TCN Epoch  28 | Train Loss 0.6642 | Val AUC 0.5349 | Val Acc 0.5239
TCN Epoch  29 | Train Loss 0.6624 | Val AUC 0.5283 | Val Acc 0.5200
TCN Epoch  30 | Train Loss 0.6601 | Val AUC 0.5285 | Val Acc 0.5218
TCN Epoch  31 | Train Loss 0.6604 | Val AUC 0.5301 | Val Acc 0.5230
TCN Epoch  32 | Train Loss 0.6560 | Val AUC 0.5367 | Val Acc 0.5267
TCN Epoch  33 | Train Loss 0.6547 | Val AUC 0.5309 | Val Acc 0.5238
TCN Epoch  34 | Train Loss 0.6511 | Val AUC 0.5380 | Val Acc 0.5266
TCN Epoch  35 | Train Loss 0.6509 | Val AUC 0.5361 | Val Acc 0.5249
TCN Epoch  36 | Train Loss 0.6494 | Val AUC 0.5385 | Val Acc 0.5285
TCN Epoch  37 | Train Loss 0.6474 | Val AUC 0.5437 | Val Acc 0.5315
TCN Epoch  38 | Train Loss 0.6452 | Val AUC 0.5364 | Val Acc 0.5281
TCN Epoch  39 | Train Loss 0.6420 | Val AUC 0.5419 | Val Acc 0.5317
TCN Epoch  40 | Train Loss 0.6408 | Val AUC 0.5390 | Val Acc 0.5291
TCN Epoch  41 | Train Loss 0.6383 | Val AUC 0.5422 | Val Acc 0.5322
TCN Epoch  42 | Train Loss 0.6344 | Val AUC 0.5423 | Val Acc 0.5317
TCN Epoch  43 | Train Loss 0.6354 | Val AUC 0.5437 | Val Acc 0.5317
TCN Epoch  44 | Train Loss 0.6321 | Val AUC 0.5426 | Val Acc 0.5312
TCN Epoch  45 | Train Loss 0.6299 | Val AUC 0.5436 | Val Acc 0.5322
TCN Epoch  46 | Train Loss 0.6275 | Val AUC 0.5416 | Val Acc 0.5318
TCN Epoch  47 | Train Loss 0.6267 | Val AUC 0.5434 | Val Acc 0.5318
TCN Epoch  48 | Train Loss 0.6236 | Val AUC 0.5463 | Val Acc 0.5347
TCN Epoch  49 | Train Loss 0.6233 | Val AUC 0.5472 | Val Acc 0.5354
TCN Epoch  50 | Train Loss 0.6176 | Val AUC 0.5408 | Val Acc 0.5297
TCN Epoch  51 | Train Loss 0.6151 | Val AUC 0.5472 | Val Acc 0.5353
TCN Epoch  52 | Train Loss 0.6169 | Val AUC 0.5477 | Val Acc 0.5353
TCN Epoch  53 | Train Loss 0.6115 | Val AUC 0.5511 | Val Acc 0.5398
TCN Epoch  54 | Train Loss 0.6102 | Val AUC 0.5497 | Val Acc 0.5386
TCN Epoch  55 | Train Loss 0.6098 | Val AUC 0.5473 | Val Acc 0.5345
TCN Epoch  56 | Train Loss 0.6082 | Val AUC 0.5448 | Val Acc 0.5338
TCN Epoch  57 | Train Loss 0.6045 | Val AUC 0.5523 | Val Acc 0.5404
TCN Epoch  58 | Train Loss 0.6029 | Val AUC 0.5501 | Val Acc 0.5390
TCN Epoch  59 | Train Loss 0.5989 | Val AUC 0.5504 | Val Acc 0.5367
TCN Epoch  60 | Train Loss 0.5963 | Val AUC 0.5517 | Val Acc 0.5367
TCN Epoch  61 | Train Loss 0.5957 | Val AUC 0.5521 | Val Acc 0.5388
TCN Epoch  62 | Train Loss 0.5930 | Val AUC 0.5544 | Val Acc 0.5423
TCN Epoch  63 | Train Loss 0.5914 | Val AUC 0.5529 | Val Acc 0.5386
TCN Epoch  64 | Train Loss 0.5899 | Val AUC 0.5565 | Val Acc 0.5410
TCN Epoch  65 | Train Loss 0.5905 | Val AUC 0.5577 | Val Acc 0.5409
TCN Epoch  66 | Train Loss 0.5882 | Val AUC 0.5524 | Val Acc 0.5390
TCN Epoch  67 | Train Loss 0.5833 | Val AUC 0.5550 | Val Acc 0.5397
TCN Epoch  68 | Train Loss 0.5816 | Val AUC 0.5576 | Val Acc 0.5422
TCN Epoch  69 | Train Loss 0.5806 | Val AUC 0.5535 | Val Acc 0.5393
TCN Epoch  70 | Train Loss 0.5786 | Val AUC 0.5571 | Val Acc 0.5419
TCN Epoch  71 | Train Loss 0.5752 | Val AUC 0.5519 | Val Acc 0.5405
TCN Epoch  72 | Train Loss 0.5769 | Val AUC 0.5527 | Val Acc 0.5397
TCN Epoch  73 | Train Loss 0.5737 | Val AUC 0.5560 | Val Acc 0.5401
TCN Epoch  74 | Train Loss 0.5715 | Val AUC 0.5524 | Val Acc 0.5392
TCN Epoch  75 | Train Loss 0.5691 | Val AUC 0.5582 | Val Acc 0.5417
TCN Epoch  76 | Train Loss 0.5694 | Val AUC 0.5563 | Val Acc 0.5414
TCN Epoch  77 | Train Loss 0.5687 | Val AUC 0.5594 | Val Acc 0.5439
TCN Epoch  78 | Train Loss 0.5640 | Val AUC 0.5585 | Val Acc 0.5424
TCN Epoch  79 | Train Loss 0.5611 | Val AUC 0.5616 | Val Acc 0.5453
TCN Epoch  80 | Train Loss 0.5617 | Val AUC 0.5583 | Val Acc 0.5428
TCN Epoch  81 | Train Loss 0.5596 | Val AUC 0.5592 | Val Acc 0.5432
TCN Epoch  82 | Train Loss 0.5577 | Val AUC 0.5589 | Val Acc 0.5456
TCN Epoch  83 | Train Loss 0.5577 | Val AUC 0.5622 | Val Acc 0.5456
TCN Epoch  84 | Train Loss 0.5578 | Val AUC 0.5600 | Val Acc 0.5450
TCN Epoch  85 | Train Loss 0.5517 | Val AUC 0.5583 | Val Acc 0.5428
TCN Epoch  86 | Train Loss 0.5556 | Val AUC 0.5615 | Val Acc 0.5458
TCN Epoch  87 | Train Loss 0.5519 | Val AUC 0.5615 | Val Acc 0.5430
TCN Epoch  88 | Train Loss 0.5501 | Val AUC 0.5629 | Val Acc 0.5436
TCN Epoch  89 | Train Loss 0.5481 | Val AUC 0.5613 | Val Acc 0.5435
TCN Epoch  90 | Train Loss 0.5474 | Val AUC 0.5634 | Val Acc 0.5442
TCN Epoch  91 | Train Loss 0.5432 | Val AUC 0.5608 | Val Acc 0.5432
TCN Epoch  92 | Train Loss 0.5441 | Val AUC 0.5626 | Val Acc 0.5455
TCN Epoch  93 | Train Loss 0.5422 | Val AUC 0.5602 | Val Acc 0.5445
TCN Epoch  94 | Train Loss 0.5402 | Val AUC 0.5645 | Val Acc 0.5463
TCN Epoch  95 | Train Loss 0.5376 | Val AUC 0.5614 | Val Acc 0.5434
TCN Epoch  96 | Train Loss 0.5319 | Val AUC 0.5605 | Val Acc 0.5447
TCN Epoch  97 | Train Loss 0.5364 | Val AUC 0.5614 | Val Acc 0.5467
TCN Epoch  98 | Train Loss 0.5332 | Val AUC 0.5630 | Val Acc 0.5467
TCN Epoch  99 | Train Loss 0.5348 | Val AUC 0.5629 | Val Acc 0.5469
TCN Epoch 100 | Train Loss 0.5316 | Val AUC 0.5627 | Val Acc 0.5449
TCN Epoch 101 | Train Loss 0.5298 | Val AUC 0.5639 | Val Acc 0.5461
TCN Epoch 102 | Train Loss 0.5288 | Val AUC 0.5614 | Val Acc 0.5470
TCN Epoch 103 | Train Loss 0.5296 | Val AUC 0.5639 | Val Acc 0.5453
TCN Epoch 104 | Train Loss 0.5273 | Val AUC 0.5660 | Val Acc 0.5482
TCN Epoch 105 | Train Loss 0.5265 | Val AUC 0.5595 | Val Acc 0.5453
TCN Epoch 106 | Train Loss 0.5260 | Val AUC 0.5659 | Val Acc 0.5513
TCN Epoch 107 | Train Loss 0.5237 | Val AUC 0.5622 | Val Acc 0.5475
TCN Epoch 108 | Train Loss 0.5233 | Val AUC 0.5619 | Val Acc 0.5455
TCN Epoch 109 | Train Loss 0.5232 | Val AUC 0.5638 | Val Acc 0.5484
TCN Epoch 110 | Train Loss 0.5227 | Val AUC 0.5635 | Val Acc 0.5470
TCN Epoch 111 | Train Loss 0.5170 | Val AUC 0.5643 | Val Acc 0.5468
TCN Epoch 112 | Train Loss 0.5190 | Val AUC 0.5625 | Val Acc 0.5464
TCN Epoch 113 | Train Loss 0.5151 | Val AUC 0.5601 | Val Acc 0.5456
TCN Epoch 114 | Train Loss 0.5177 | Val AUC 0.5639 | Val Acc 0.5493
TCN Epoch 115 | Train Loss 0.5140 | Val AUC 0.5633 | Val Acc 0.5482
TCN Epoch 116 | Train Loss 0.5140 | Val AUC 0.5639 | Val Acc 0.5488
TCN Epoch 117 | Train Loss 0.5111 | Val AUC 0.5655 | Val Acc 0.5498
TCN Epoch 118 | Train Loss 0.5118 | Val AUC 0.5672 | Val Acc 0.5514
TCN Epoch 119 | Train Loss 0.5099 | Val AUC 0.5629 | Val Acc 0.5473
TCN Epoch 120 | Train Loss 0.5073 | Val AUC 0.5621 | Val Acc 0.5474
TCN Epoch 121 | Train Loss 0.5074 | Val AUC 0.5632 | Val Acc 0.5485
TCN Epoch 122 | Train Loss 0.5099 | Val AUC 0.5638 | Val Acc 0.5474
TCN Epoch 123 | Train Loss 0.5032 | Val AUC 0.5648 | Val Acc 0.5470
TCN Epoch 124 | Train Loss 0.5017 | Val AUC 0.5640 | Val Acc 0.5484
TCN Epoch 125 | Train Loss 0.5002 | Val AUC 0.5662 | Val Acc 0.5489
TCN Epoch 126 | Train Loss 0.5037 | Val AUC 0.5675 | Val Acc 0.5492
TCN Epoch 127 | Train Loss 0.4999 | Val AUC 0.5652 | Val Acc 0.5477
TCN Epoch 128 | Train Loss 0.4998 | Val AUC 0.5649 | Val Acc 0.5501
TCN Epoch 129 | Train Loss 0.5009 | Val AUC 0.5654 | Val Acc 0.5489
TCN Epoch 130 | Train Loss 0.5002 | Val AUC 0.5655 | Val Acc 0.5501
TCN Epoch 131 | Train Loss 0.4964 | Val AUC 0.5633 | Val Acc 0.5484
TCN Epoch 132 | Train Loss 0.4978 | Val AUC 0.5673 | Val Acc 0.5489
TCN Epoch 133 | Train Loss 0.4966 | Val AUC 0.5683 | Val Acc 0.5512
TCN Epoch 134 | Train Loss 0.4916 | Val AUC 0.5676 | Val Acc 0.5504
TCN Epoch 135 | Train Loss 0.4916 | Val AUC 0.5650 | Val Acc 0.5475
TCN Epoch 136 | Train Loss 0.4907 | Val AUC 0.5634 | Val Acc 0.5476
TCN Epoch 137 | Train Loss 0.4953 | Val AUC 0.5673 | Val Acc 0.5483
TCN Epoch 139 | Train Loss 0.4907 | Val AUC 0.5645 | Val Acc 0.5488
TCN Epoch 140 | Train Loss 0.4902 | Val AUC 0.5694 | Val Acc 0.5532
TCN Epoch 141 | Train Loss 0.4892 | Val AUC 0.5662 | Val Acc 0.5523
TCN Epoch 142 | Train Loss 0.4871 | Val AUC 0.5680 | Val Acc 0.5503
TCN Epoch 143 | Train Loss 0.4915 | Val AUC 0.5703 | Val Acc 0.5514
TCN Epoch 144 | Train Loss 0.4829 | Val AUC 0.5675 | Val Acc 0.5503
TCN Epoch 145 | Train Loss 0.4810 | Val AUC 0.5672 | Val Acc 0.5503
TCN Epoch 146 | Train Loss 0.4816 | Val AUC 0.5670 | Val Acc 0.5501
TCN Epoch 147 | Train Loss 0.4826 | Val AUC 0.5661 | Val Acc 0.5503
TCN Epoch 148 | Train Loss 0.4796 | Val AUC 0.5682 | Val Acc 0.5523
TCN Epoch 149 | Train Loss 0.4769 | Val AUC 0.5658 | Val Acc 0.5499
TCN Epoch 150 | Train Loss 0.4785 | Val AUC 0.5655 | Val Acc 0.5491
TCN Epoch 151 | Train Loss 0.4784 | Val AUC 0.5668 | Val Acc 0.5494
TCN Epoch 152 | Train Loss 0.4788 | Val AUC 0.5679 | Val Acc 0.5496
TCN Epoch 153 | Train Loss 0.4774 | Val AUC 0.5682 | Val Acc 0.5498
TCN Epoch 154 | Train Loss 0.4783 | Val AUC 0.5642 | Val Acc 0.5486
TCN Epoch 155 | Train Loss 0.4769 | Val AUC 0.5664 | Val Acc 0.5518
TCN Epoch 156 | Train Loss 0.4727 | Val AUC 0.5676 | Val Acc 0.5503
TCN Epoch 157 | Train Loss 0.4706 | Val AUC 0.5683 | Val Acc 0.5505
TCN Epoch 158 | Train Loss 0.4682 | Val AUC 0.5685 | Val Acc 0.5527
TCN Epoch 159 | Train Loss 0.4730 | Val AUC 0.5691 | Val Acc 0.5514
TCN Epoch 160 | Train Loss 0.4678 | Val AUC 0.5692 | Val Acc 0.5528
TCN Epoch 161 | Train Loss 0.4707 | Val AUC 0.5685 | Val Acc 0.5528
TCN Epoch 162 | Train Loss 0.4699 | Val AUC 0.5681 | Val Acc 0.5529
TCN Epoch 163 | Train Loss 0.4697 | Val AUC 0.5708 | Val Acc 0.5523
TCN Epoch 164 | Train Loss 0.4684 | Val AUC 0.5692 | Val Acc 0.5524
TCN Epoch 165 | Train Loss 0.4663 | Val AUC 0.5663 | Val Acc 0.5503
TCN Epoch 166 | Train Loss 0.4675 | Val AUC 0.5696 | Val Acc 0.5538
TCN Epoch 167 | Train Loss 0.4685 | Val AUC 0.5680 | Val Acc 0.5515
TCN Epoch 168 | Train Loss 0.4681 | Val AUC 0.5712 | Val Acc 0.5535
TCN Epoch 169 | Train Loss 0.4645 | Val AUC 0.5682 | Val Acc 0.5507
TCN Epoch 170 | Train Loss 0.4607 | Val AUC 0.5685 | Val Acc 0.5515
TCN Epoch 171 | Train Loss 0.4617 | Val AUC 0.5681 | Val Acc 0.5506
TCN Epoch 172 | Train Loss 0.4593 | Val AUC 0.5692 | Val Acc 0.5518
TCN Epoch 173 | Train Loss 0.4630 | Val AUC 0.5706 | Val Acc 0.5544
TCN Epoch 174 | Train Loss 0.4576 | Val AUC 0.5705 | Val Acc 0.5534
"""

# Regex to extract metrics:
# Matches: Epoch   1 | Train Loss 0.7494 | Val AUC 0.5227 | Val Acc 0.5140
pattern = re.compile(
    r"TCN Epoch\s+(\d+)\s+\|\s+Train Loss\s+([\d.]+)\s+\|\s+Val AUC\s+([\d.]+)\s+\|\s+Val Acc\s+([\d.]+)"
)

epochs = []
train_losses = []
val_aucs = []
val_accs = []

for match in pattern.findall(log_text):
    epoch, loss, auc, acc = match
    epochs.append(int(epoch))
    train_losses.append(float(loss))
    val_aucs.append(float(auc))
    val_accs.append(float(acc))

# Build DataFrame
df = pd.DataFrame({
    "Epoch": epochs,
    "Train Loss": train_losses,
    "Val AUC": val_aucs,
    "Val Acc": val_accs
})

print(df.head())
print("\nExtracted", len(df), "epochs.")

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss")
plt.plot(df["Epoch"], df["Val AUC"], label="Val AUC")
plt.plot(df["Epoch"], df["Val Acc"], label="Val Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Training Metrics Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
