import math
from collections import Counter, defaultdict
from functools import partial
inputs = [
    ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'no'}, False),
    ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'yes'}, False),
    ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'R', 'tweets': 'yes', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'R', 'tweets': 'yes', 'phd': 'yes'}, False),
    ({'level': 'Mid', 'lang': 'R', 'tweets': 'yes', 'phd': 'yes'}, True),
    ({'level': 'Senior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, False),
    ({'level': 'Senior', 'lang': 'R', 'tweets': 'yes', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'no'}, True),
    ({'level': 'Senior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'yes'}, True),
    ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, True),
    ({'level': 'Mid', 'lang': 'Java', 'tweets': 'yes', 'phd': 'no'}, True),
    ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, False)
]
def entropy(class_probabilites):
    # 클래스에 속할 확률을 입력하면 엔트로피 계산
    # 확률이 0인 경우는 제외함
    return sum(-p * math.log(p, 2) for p in class_probabilites if p is not 0)

def class_probabilities(labels):
    # 레이블의 총 개수 계산 : ex) 5
    total_count = len(labels)
    # Counter(labels) = {Class0 : 3, Class1 : 2}
    # class0 prob = 0.6, class1 prob = 0.4 반환
    return [float(count) / float(total_count) for count in Counter(labels).values()]

def data_entropy(labeled_data):
    # 데이터를 받아서 레이블 정보만 뺀 뒤 리스트로 저장
    # ex) labels = [0, 0, 0, 1, 1]
    labels = [label for _, label in labeled_data]
    # 클래스 비율 계산
    probabilities = class_probabilities(labels)
    # 클래스 비율을 토대로 엔트로피 계산
    return entropy(probabilities)

def partition_entropy(subsets):
    # subset은 레이블이 있는 데이터의 list의 list
    # 그에 대한 엔트로피를 계산한 뒤 모든 subset의 엔트로피 합친 값 반환
    total_count = sum(len(subset) for subset in subsets)
    # subset A의 엔트로피는 A 요소별 엔트로피의 합 * A의 영역 비율
    return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)

def partition_by(inputs, attribute):
    # attribute 기준으로 inputs를 부분 집합으로 분리
    # attribute 변수 내에 3개 값이 있다면 그룹수 = 3
    # ex) level 기준 = Senior, Mid, Junior 3개 그룹
    groups = defaultdict(list)
    for input in inputs:
        # 특정 attribute의 값을 불러옴
        key = input[0][attribute]
        # 이 input을 올바른 list에 추가
        groups[key].append(input)
    return groups

def partition_entropy_by(inputs, attribute):
    # 주어진 파티션에 대응되는 엔트로피를 계산
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())

def build_tree(inputs, split_candidates=None):
    # 첫 분기라면 입력 데이터의 모든 변수가 분기 후보
    if split_candidates is None:
        # 'lang', 'tweets', 'phd', 'level' 모두 후보
        split_candidates = inputs[0][0].keys()

    # 입력 데이터에서 범주별 개수를 세어 본다
    num_inputs = len(inputs)
    num_class0 = len([label for _, label in inputs if label])
    num_class1 = num_inputs - num_class0

    # class0(true)이 하나도 없으면 False leaf 반환
    if num_class0 == 0: return False
    # class1(false)이 하나도 없으면 Ture leaf 반환
    if num_class1 == 0: return True

    # 파티션 기준으로 사용할 변수가 없다면
    if not split_candidates:
        # 다수결로 결정
        # class0(true)가 많으면 true,
        # class1(false)가 많으면 false 반환
        return num_class0 >= num_class1

    # 아니면 가장 적합한 변수를 기준으로 분기
    best_attribute = min(split_candidates,
                         key=partial(partition_entropy_by, inputs))
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates
                      if a != best_attribute]

    # 재귀적으로 서브트리를 구축
    subtrees = { attribute_value : build_tree(subset, new_candidates)
                 for attribute_value, subset in partitions.items()}
    # 기본값
    subtrees[None] = num_class0 > num_class1 
    return (best_attribute, subtrees)

def classify(tree, input):
    # 주어진 tree를 기준으로 input을 분류
    # 잎 노드이면 값 반환
    if tree in [True, False]:
        return tree

    # 그게 아니면 데이터의 변수로 분기
    # 키로 변수값, 값으로 서브트리를 나타내는 dict 사용
    attribute, subtree_dict = tree

    # 만약 입력된 데이터 변수 가운데 하나가
    # 기존에 관찰되지 않았다면 None
    subtree_key = input.get(attribute)

    # 키에 해당하는 서브트리가 존재하지 않을 때
    if subtree_key not in subtree_dict:
        # None 서브트리를 사용
        subtree_key = None

    # 적절한 서브트리를 선택
    subtree = subtree_dict[subtree_key]
    # 그리고 입력된 데이터를 분류
    return classify(subtree, input)

tree = build_tree(inputs)
print(classify(tree,
        { "level" : "Junior",
          "lang" : "Java",
          "tweets" : "yes",
          "phd" : "no"} )) # -> True
print(classify(tree,
        { "level" : "Junior",
          "lang" : "Java",
          "tweets" : "yes",
          "phd" : "yes"} )) # -> False