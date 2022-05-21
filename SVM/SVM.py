from libsvm.commonutil import svm_read_problem
from libsvm.svmutil import svm_train, svm_predict

if __name__ == '__main__':
    y, x = svm_read_problem('ex4x.txt')
    yt, xt = svm_read_problem('ex4x.txt')
    
    # t 为0 默认线性核函数
    model = svm_train(y, x, '-c 1 -t 0')

    # 使用样本来预测
    p_label, p_acc, p_val = svm_predict(yt, xt, model)
    print(p_label)



