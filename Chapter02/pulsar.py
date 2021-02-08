import numpy as np
import csv
import time
# from Chapter01.abalone import init_model, train_and_test, arrange_data, get_train_data, get_test_data, run_train, run_test, forward_neuralnet, backprop_neuralnet
# abalone.py에서 불러와 사용하려했는데, global 선언에서 error 발생

# 2021.2.8
# binary classification

# 코드 재사용
# parameter 초기화
def init_model():
    global weight, bias, input_cnt, output_cnt     # 다른 함수에서 이용가능하도록 전역변수로 선언 -> 괜찮을까?
    weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt])     # [10, 1]
    bias = np.zeros([output_cnt])     # [1]


# 학습 및 평가 함수 정의
def train_and_test(epochs, batch_size, report):
    step_size = arrange_data(batch_size)    # 배치당 step size계산, 데이터 shuffle, train, test분리
    test_x, test_y = get_test_data()

    for epoch in range(epochs):
        losses, accs = [], []
        # epoch당 step수 (train data 수 / mini_batch 수)
        for n in range(step_size):
            train_x, train_y = get_train_data(batch_size, n)
            loss, acc = run_train(train_x, train_y)
            losses.append(loss)
            accs.append(acc)

        if report > 0 and (epoch+1) % report == 0:
            acc = run_test(test_x, test_y)
            print("EPOCH {}: LOSS={:5.3f}, ACC={:5.3f}/{:5.3f}".\
                  format(epoch+1, np.mean(losses), np.mean(accs), acc))    # np.mean(accs) : 매 epoch시 test acc의 평균값, acc : 직전 epoch의 test acc값

    final_acc = run_test(test_x, test_y)
    print('\nFinal Test : final accuracy = {:5.3f}'.format(final_acc))

# dataset 처리, step_size 생성
def arrange_data(batch_size):
    global data, shuffle_map, test_begin_idx
    shuffle_map = np.arange(data.shape[0])   # 데이터 수 만큼 일련번호 발생
    np.random.shuffle(shuffle_map)           # 무작위로 순서 섞음
    step_size = int(data.shape[0]*0.8)  // batch_size   # train data 수 / batch_size
    test_begin_idx = step_size * batch_size      # test data의 시작지점 index
    return step_size

def get_test_data():
    global data, shuffle_map, test_begin_idx, output_cnt
    test_data = data[shuffle_map[test_begin_idx:]]     # test data 생성
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:]   # test_x(입력 벡터), test_y(정답 벡터)

def get_train_data(batch_size, nth):
    global data, shuffle_map, test_begin_idx, output_cnt
    if nth == 0:   # 전체 data를 한번에 training
        np.random.shuffle(shuffle_map[:test_begin_idx])     # train data 생성
    train_data = data[shuffle_map[batch_size*nth:batch_size*(nth+1)]]  # nth -> epoch idx, nth : 첫 번째 epoch, nth는 step_size를 통해 n으로 전달 위 코드 참고
    return train_data[:, :-output_cnt], train_data[:, -output_cnt:]    # train_x, train_y

# per step process
def run_train(x, y):
    output, aux_nn = forward_neuralnet(x)            # 순전파, 신경망 출력 결과
    loss, aux_pp = forward_postproc(output, y)       # 순전파, loss 계산(predict과 ground truth 차이 오차)
    accuracy = eval_accuracy(output, y)              # 정확도 계산

    G_loss = 1.0   # Loss에 대한 loss의 편미분 값 1 -> 역전파의 시작점
    G_output = backprop_postproc(G_loss, aux_pp)     # 역전파
    backprop_neuralnet(G_output, aux_nn)               # 역전파

    return loss, accuracy

# per step process
def run_test(x, y):
    output, _ = forward_neuralnet(x)
    accuracy = eval_accuracy(output, y)
    return accuracy


# 순전파 및 역전파 함수 정의
def forward_neuralnet(x):
    global weight, bias
    output = np.matmul(x, weight) + bias
    return output, x

def backprop_neuralnet(G_output, x):
    global weight, bias
    g_output_w = x.transpose()
    G_w = np.matmul(g_output_w, G_output)
    G_b = np.sum(G_output, axis=0)

    weight -= LERNING_RATE * G_w
    bias -= LERNING_RATE * G_b

# main 함수 정리
def pulsar_exec(num_epochs=10, batch_size=10, report=1):
    load_pulsar_dataset()
    init_model()
    train_and_test(num_epochs, batch_size, report)

# dataread
def load_pulsar_dataset():
    with open("../Data/pulsar_stars.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)

    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 8, 1     # 입출력 벡터 크기
    data =np.asarray(rows, dtype='float32')    # list구조는 numpy에서 제공하는 연산에 부적절, 비효율적 -> 배열구조로 변환


# 후처리 과정에 대한 순전파와 역전파 함수의 재정의

# loss 계산
def forward_postproc(output, y):
    entropy = sigmoid_cross_entropy_with_logits(y, output)
    loss = np.mean(entropy)
    return loss, [y, output, entropy]

# loss 기울기 계산, 순전파의 역순 -> G_output
def backprop_postproc(G_loss, aux):
    y, output, entropy = aux

    g_loss_entropy = 1.0 / np.prod(entropy.shape)  # np.prod -> 배열 원소간 곱(axis 기준), G_loss인 1.0을 행렬의 원소 수로 나누어 각 원소의 손실 기울기로 부여
    g_entropy_output = sigmoid_cross_entropy_with_logits_derv(y, output)  # y, output의 각 원소싸엥 대해 sigmoid cross entropy의 편미분 값 구함 (entropy와 output사이의 부분 기울기)

    G_entropy = g_loss_entropy * G_loss
    G_output = g_entropy_output * G_entropy

    return G_output

# 이진 분류 정확도 함수
def eval_accuracy(output, y):
    """
    np.greater([4,2],[2,2])
    array([ True, False])
    """
    estimate = np.greater(output, 0)   # 0보다 크면 True (해당하는 array,list 원소 별)
    answer = np.greater(y, 0.5)        # 0.5보다 크면 True (해당하는 array,list 원소 별)
    correct = np.equal(estimate, answer)   # True, False (해당하는 array,list 원소 별)

    return np.mean(correct)   # True 비율 구해짐 (python은 참은 1, 거짓은 0이라 간주)

# 활성화 함수 -> class로 정의하고 forward,backward 합치면 더 좋지 않을까?
def relu(x):
    return np.maximum(x, 0)

# https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits?hl=ko 참고
def sigmoid(x):
    return np.exp(-relu(-x)) / (1.0 + np.exp(-np.abs(x)))  # 왜 relu가 들어가지? -> overflow 방지

# sigmoid 편미분
def sigmoid_derv(x, y):
    return y * (1-y)

# z(ground truth) -> y, x -> output
def sigmoid_cross_entropy_with_logits(z, x):
    return relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))

def sigmoid_cross_entropy_with_logits_derv(z, x):
    return -z + sigmoid(x)


if __name__ == "__main__":
    print("TRAIN START!")
    np.random.seed(1234)


    def randomize(): np.random.seed(time.time())  # 난수 함수 초기화

    # hyperparameter 정의
    RND_MEAN = 0  # 정규분포 난수값의 평균
    RND_STD = 0.0030  # 표준편차
    LERNING_RATE = 0.1

    num_epochs = 10
    batch_size = 10
    report = 1
    pulsar_exec(num_epochs, batch_size, report)






