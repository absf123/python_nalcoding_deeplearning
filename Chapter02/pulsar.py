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
        losses = []
        # epoch당 step수 (train data 수 / mini_batch 수)
        for n in range(step_size):
            train_x, train_y = get_train_data(batch_size, n)
            loss, _ = run_train(train_x, train_y)
            losses.append(loss)
            # accs.append(acc)

        if report > 0 and (epoch+1) % report == 0:
            acc = run_test(test_x, test_y)
            acc_str = ','.join(['%5.3f']*4) % tuple(acc)        # eval_accuracy의 출력 결과 수와 포멧에 맞게 문자열로 수정
            print("EPOCH {}: LOSS={:5.3f}, RESULT={}".\
                  format(epoch+1, np.mean(losses), acc_str))    # np.mean(accs) : 매 epoch시 test acc의 평균값, acc : 직전 epoch의 test acc값

    acc = run_test(test_x, test_y)
    acc_str = ','.join(['%5.3f']*4) % tuple(acc)
    print('\nFinal Test : final result = {}'.format(acc_str))


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
    weight -= LEARNING_RATE * G_w
    bias -= LEARNING_RATE * G_b

# main 함수 정리
def pulsar_exec(num_epochs=10, batch_size=10, report=1, adjust_ratio=False):
    load_pulsar_dataset(adjust_ratio)   # adjust_ratio : True(별과 pulsar비율 맞춤), False(기존 비율 그대로 사용)
    init_model()
    train_and_test(num_epochs, batch_size, report)


# dataread
def load_pulsar_dataset(adjust_ratio):
    pulsars, stars = [], []    # 펄서와 별을 각각 나누어 리스트에 저장
    with open("../Data/pulsar_stars.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        rows = []
        for row in csvreader:
            if row[8] == '1': pulsars.append(row)
            else : stars.append(row)

    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 8, 1     # 입출력 벡터 크기

    star_cnt, pulsar_cnt = len(stars), len(pulsars)

    if adjust_ratio:
        # true
        data = np.zeros([2*star_cnt, 9])   # pulsar를 star수 만큼 늘릴거라 star_cnt의 두배만큼 설정
        data[0:star_cnt, :] = np.asarray(stars, dtype='float32')
        for n in range(star_cnt):
            data[star_cnt+n] = np.asarray(pulsars[n%pulsar_cnt], dtype='float32')
    else:
        # false
        data = np.zeros([star_cnt+pulsar_cnt, 9])
        data[0:star_cnt, :] = np.asarray(stars, dtype='float32')
        data[star_cnt:, :] = np.asarray(pulsars, dtype='float32') # list구조는 numpy에서 제공하는 연산에 부적절, 비효율적 -> 배열구조로 변환


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
    est_yes = np.greater(output, 0)   # 0보다 크면 True (해당하는 array,list 원소 별)
    ans_yes = np.greater(y, 0.5)        # 0.5보다 크면 True (해당하는 array,list 원소 별)
    est_no = np.logical_not(est_yes)
    ans_no = np.logical_not(ans_yes)

    TP = np.sum(np.logical_and(est_yes, ans_yes))
    FP = np.sum(np.logical_and(est_yes, ans_no))
    FN = np.sum(np.logical_and(est_no, ans_yes))
    TN = np.sum(np.logical_and(est_no, ans_no))

    accuracy = safe_div(TP+TN, TP+FP+FN+TN)
    precision = safe_div(TP, TP+FP)
    recall = safe_div(TP, TP+FN)
    f1 = 2 * safe_div(precision*recall, precision+recall)

    # return np.mean(correct)   # True 비율 구해짐 (python은 참은 1, 거짓은 0이라 간주)
    return [accuracy, precision, recall, f1]


def safe_div(p, q):
    p, q = float(p), float(q)  # type 오류 방지
    if np.abs(q) < 1.0e-20: return np.sign(p)  # 0으로 나눠지는 것을 방지
    return p / q


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
    np.random.seed(42)


    def randomize(): np.random.seed(time.time())  # 난수 함수 초기화

    # hyperparameter 정의
    # RND_MEAN = 0  # 정규분포 난수값의 평균
    # RND_STD = 0.0030  # 표준편차
    # LERNING_RATE = 0.2

    RND_MEAN = 0
    RND_STD = 0.0030

    LEARNING_RATE = 0.001

    num_epochs = 10
    batch_size = 10
    report = 1

    # Loss 값도 많이 다르고 acc가 불안정함 -> 이유를 모르겠음 -> loss값은 learning_rate 차이였음
    pulsar_exec(num_epochs, batch_size, report, adjust_ratio=True)






