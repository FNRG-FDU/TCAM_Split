import numpy as np
import math


class Rule:
    def __init__(self, src_b, src_e, dst_b, dst_e, prio):
        self.src = [src_b, src_e]
        self.dst = [dst_b, dst_e]
        self.prio = prio
        self.taddr = 0


class TCAM:
    def __init__(self):
        self.cur_num = 0  # 存储当前规则数
        self.move = 0  # 存储当前总移动次数
        self.rule_set = []  # 存储规则本身
        self.graph = []  # 存储规则序号
        self.ram = []  # 存储规则序号，模拟TCAM结构
        self.state = [0, 0, 0, 0, 0, 0]

    def overlap(self, r1, r2):
        if max(r1.src[0], r2.src[0]) > min(r1.src[1], r2.src[1]):
            return False
        if max(r1.dst[0], r2.dst[0]) > min(r1.dst[1], r2.dst[1]):
            return False
        return True

    def build(self,r):
        son = []
        uaddr = -1
        daddr = 30000  # 任意大数
        for i in range(0, len(self.graph)):
            if self.overlap(self.rule_set[i], r):
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

    def insert(self, r):
        a = self.move
        self.rule_set.append(r)
        [uaddr, daddr] = self.build(r)
        self._insert(uaddr, daddr, self.cur_num)

        self.state[0] = (self.state[0] * (self.cur_num - 1) + r.src[1] - r.src[0])/self.cur_num
        self.state[1] = (self.state[1] * (self.cur_num - 1) + r.dst[1] - r.dst[0]) / self.cur_num
        self.state[2] = (self.state[2] * (self.cur_num - 1) + (r.src[1] + r.src[0])/2) / self.cur_num
        self.state[3] = (self.state[3] * (self.cur_num - 1) + (r.dst[1] + r.dst[0])/2) / self.cur_num
        self.state[4] = (self.state[4] * (self.cur_num - 1) + r.prio) / self.cur_num
        self.state[5] = self.cur_num

        return self.move - a

    def get_state(self, r):
        # 状态各维度含义：[（当前TCAM）平均长度，平均宽度，平均x0，平均y0，平均优先级，规则数，（被依赖规则部分）规则数，（依赖规则部分）规则数，]
        cnu = 0
        cnd = 0
        for i in range(0, len(self.graph)):
            if self.overlap(self.rule_set[i], r):
                if self.rule_set[i].prio > r.prio:
                    cnd += 1
                else:
                    cnu += 1
        # res = self.state[:]
        res = []
        res.append(cnd)
        res.append(cnu)
        return res


if __name__ == "__main__":
    tcam1 = TCAM()
    tcam2 = TCAM()

    rule_set = []
    f = open("./data/fw1.txt")
    f1 = open("./datares/c-fw1", "w")
    s = f.readlines()

    for line in s:
        ss = line.split('  ')
        rule_set.append(Rule(int(ss[0]), int(ss[1]), int(ss[2]), int(ss[3]), int(ss[8])))
    for i in range(0, len(rule_set)):
        r = rule_set[i]
        state1 = tcam1.get_state(r)
        state2 = tcam2.get_state(r)
        state = state1 + state2 + [r.src[1] - r.src[0], r.dst[1] - r.dst[0], (r.src[1]+r.src[0])/2, (r.dst[1]+r.dst[0])/2, r.prio]
        flag = i % 2  # 此处用决策函数替代
        if i == 800:
            c = 1
        if flag == 0:
            num = tcam1.insert(r)
            print("TCAM1:", num)
            f1.write("TCAM1:" + str(num))
        else:
            num = tcam2.insert(r)
            print("TCAM2:", num)
            f1.write("TCAM2:" + str(num))
    print("total cost:", tcam1.move, "+", tcam2.move, "=", tcam1.move + tcam2.move)
    f1.write("total cost:" + str(tcam1.move) + "+" + str(tcam2.move) + "=" + str(tcam1.move + tcam2.move))
    print("rule_num1:", tcam1.cur_num, "rule_num2:", tcam2.cur_num)
    f1.write("rule_num1:" + str(tcam1.cur_num) + " " + "rule_num2:" + str(tcam2.cur_num))
    # print(tcam1.Over_metrix)
    # print(tcam1.Max_metrix)
    # print(tcam1.Min_metrix)

