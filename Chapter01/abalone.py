import numpy as np
import csv   # pandas는?
import time


# main function
def abalone_exec(epochs=10, batch_size=10, report=1):
    load_abalone_dataset()      # dataset load
    init_model()                # model parameter 초기화
    train_and_test(epochs, batch_size, report)              # 딥러닝 model 생성 및 전체학습과정 처리


# dataset load
def load_abalone_dataset():
    with open("../Data/abalone.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)       # 파일의 첫 행을 읽지 않고 건너뜀
        rows = []
        for row in csvreader:       # rows list에 feature 정보 수집
            rows.append(row)

    global data, input_cnt, output_cnt    # 전역변수.... -> 고치는게 좋지 않을까?, 계속 사용하기위함이라긴 함
    input_cnt, output_cnt = 10, 1         # 입출력 벡터 크기 10, 1로 선언
    data = np.zeros([len(rows), input_cnt+output_cnt])   # data 행렬 : 입출력 벡터 정보를 저장할 행렬 (input feature에 output_cnt 만큼 column 추가)

    # one-hot 인코딩
    for n, row in enumerate(rows):
        if row[0] == 'I' : data[n, 0] = 1
        if row[1] == 'M' : data[n, 1] = 1
        if row[2] == 'F' : data[n, 2] = 1
        data[n, 3:] = row[1:]

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

# 후처리 과정에 대한 순전파 및 역전파 함수 정의
def forward_postproc(output, y):
    diff = output - y
    loss = np.mean(np.square(diff))
    return loss, diff

# 순전파의 역방향
def backprop_postproc(G_loss, diff):
    shape = diff.shape

    g_loss_square= np.ones(shape) / np.prod(shape)   # np.prod -> 배열 원소간 곱(axis 기준)
    g_square_diff = 2 * diff   # diff^2 -> 2diff
    g_diff_output = 1   # diff = output - y 의 ddiff / doutput = 1

    G_square = g_loss_square * G_loss   # G_loss = 1.0
    G_diff = g_square_diff * G_square
    G_output = g_diff_output * G_diff

    return G_output

# 정확도 계산
def eval_accuracy(output, y):
    mdiff = np.mean(np.abs((output - y)/y))  # 오류율 평균
    return 1 - mdiff




if __name__ == "__main__":
    print("TRAIN START!")
    np.random.seed(1234)


    def randomize(): np.random.seed(time.time())  # 난수 함수 초기화


    # hyperparameter 정의
    RND_MEAN = 0  # 정규분포 난수값의 평균
    RND_STD = 0.0030  # 표준편차
    LERNING_RATE = 0.1


    epochs = 100              # epoch 수
    batch_size = 100          # 미니배치 수
    report = 20            # 결과 출력 주기

    abalone_exec(epochs=epochs, batch_size=batch_size, report=report)
    print()
    print(weight)
    print(bias)

    """
    [[ 0.28084865]  weight
     [-0.00357293]
     [ 0.00429812]
     [ 2.33109724]
     [ 1.85242646]
     [ 0.68413302]
     [ 2.80611213]
     [ 0.70162155]
     [ 0.56256811]
     [ 1.14419111]]
     
    [4.72213148]  bias
    """