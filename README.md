
## Doing anomaly detection.

Running the code with different options

```
python3 main.py <gan, bigan> <mnist, kdd> run --nb_epochs=<number_epochs> --label=<0, 1, 2, 3, 4, 5, 6, 7, 8, 9> --w=<float between 0 and 1> --m=<'cross-e','fm'> --d=<int> --rd=<int>
```
To reproduce the results of the paper, please use w=0.1 (as in the original AnoGAN paper which gives a weight of 0.1 to the discriminator loss), d=1 for the feature matching loss.  
