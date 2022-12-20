from config import args
from code.train import train
import time


start = time.time()
# change here
train(args)


#
end =time.time()

print(f"Elapsed time: {end-start}")
