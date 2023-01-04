
import train


if __name__ == '__main__':
    a = train.gen_trainingset(1)   
    for i in range(0,len(a[0])):
        print(a[0][i],'\n')
        print(a[1][i],'\n')
        print(a[2][i],'\n')
