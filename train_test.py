
import train
import time
if __name__ == '__main__':
    
    start = time.time()
    print("hello")
    a = train.gen_trainingset_pool(8)   

    print(len(a[0]))
    end = time.time()
    print(end - start)
