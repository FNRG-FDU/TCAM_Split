import numpy as np
import math

grain = 16  # grain size


class Rule:
    """
    This class is denoted as a rule
    """
    def __init__(self, src_b: int, src_e: int, dst_b: int, dst_e: int, prio: int):
        """
        Initialization.
        :param src_b: begin source ip
        :param src_e: end source ip
        :param dst_b: begin destination ip
        :param dst_e: end destination ip
        :param prio: priority
        """
        self.src = (src_b, src_e)
        self.dst = (dst_b, dst_e)
        self.prio = prio
        self.taddr = 0


def overlap(r1: Rule, r2: Rule):
    """
    if two rules are overlap?
    :param r1: rule #1
    :param r2: rule #2
    :return: true or false
    """
    if max(r1.src[0], r2.src[0]) > min(r1.src[1], r2.src[1]):
        return False
    if max(r1.dst[0], r2.dst[0]) > min(r1.dst[1], r2.dst[1]):
        return False
    return True


class TCAM:
    """
    This class is denoted as a TCAM
    """
    def __init__(self):
        self.cur_num = 0  # the number of rules in this TCAM
        self.move = 0  # total moving number
        self.rule_set = []  # all rules
        self.graph = []  # 存储规则序号
        self.ram = []  # 存储规则序号，模拟TCAM结构
        self.overlap_metrix = np.zeros((grain, grain))  # 重叠数矩阵
        self.side = pow(2, 32) / grain

    def build(self,r):
        son = []
        uaddr = -1
        daddr = 30000  # 任意大数
        for i in range(0, len(self.graph)):
            if overlap(self.rule_set[i], r):
                if self.rule_set[i].prio > r.prio:
                    son.append(i)
                    daddr = min(self.rule_set[i].taddr, daddr)
                else:
                    self.graph[i].append(len(self.graph))
                    uaddr = max(self.rule_set[i].taddr, uaddr)
        self.graph.append(son)
        return [uaddr, daddr]

    def _insert(self, uaddr, daddr, idx):
        # idx 为规则在rule_set中的编号
        self.move += 1
        if daddr == 30000:  # 搜索下界
            for i in self.graph[idx]:
                daddr = min(daddr, self.rule_set[i].taddr)
        if daddr == 30000:  # 无下界
            self.ram.append(idx)
            self.rule_set[idx].taddr = self.cur_num
            self.cur_num += 1
        else:
            new_idx = self.ram[daddr]
            temp = self.rule_set[new_idx].taddr
            self.rule_set[idx].taddr = temp
            self.ram[temp] = idx
            self._insert(daddr, 30000, new_idx)

    def insert(self, r: Rule):
        a = self.move
        self.rule_set.append(r)
        [uaddr, daddr] = self.build(r)
        self._insert(uaddr, daddr, self.cur_num)
        left = math.floor(r.src[0] / self.side)
        right = math.floor(r.src[1] / self.side)
        down = math.floor(r.dst[0] / self.side)
        up = math.floor(r.dst[1] / self.side)
        for i in range(left, right + 1):
            for j in range(down, up + 1):
                self.overlap_metrix[i][j] += 1
        return self.move - a

if __name__ == "__main__":
    rule_set = []
    f = open("./data/acl1.txt")
    f1 = open("./datares/a-acl1.txt", "w")
    s = f.readlines()
    tcam1 = TCAM()
    tcam2 = TCAM()
    for line in s:
        ss = line.split('  ')
        rule_set.append(Rule(int(ss[0]), int(ss[1]), int(ss[2]), int(ss[3]), int(ss[8])))
    for i in range(0, len(rule_set)):
        r = rule_set[i]
        if i % 2:
            num = tcam1.insert(r)
            print("TCAM1:", num)
            f1.write("TCAM1:" + str(num))
        else:
            num = tcam2.insert(r)
            print("TCAM2:", num)
            f1.write("TCAM2:" + str(num))
    print("total cost:", tcam1.move, "+", tcam2.move, "=", tcam1.move + tcam2.move)
    f1.write("total cost:" + str(tcam1.move) + "+" + str(tcam2.move) + "=" + str(tcam1.move + tcam2.move))


    print(tcam1.overlap_metrix)

