import numpy as np
import math


class Rule:
    def __init__(self, src_b, src_e, dst_b, dst_e, prio):
        self.src = [src_b, src_e]
        self.dst = [dst_b, dst_e]
        self.prio = prio
        self.taddr = 0


class Tcam:
    def __init__(self):
        self.cur_num = 0  # 存储当前规则数
        self.move = 0  # 存储当前总移动次数
        self.rule_set = []  # 存储规则本身
        self.graph = []  # 存储规则序号
        self.ram = []  # 存储规则序号，模拟TCAM结构
        self.state = []

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

        return self.move - a

    def get_state(self, r):
        # 状态各维度含义：[（被依赖规则部分）平均长度，平均宽度，平均x0，平均y0，平均优先级，规则数，（当前规则）长度，宽度，x0，y0，优先级，（依赖规则部分）平均长度，平均宽度，平均x0，平均y0，平均优先级，规则数，]
        up = [0, 0, 0, 0, 0, 0]
        cnu = 0
        down = [0, 0, 0, 0, 0, 0]
        cnd = 0
        for i in range(0, len(self.graph)):
            if self.overlap(self.rule_set[i], r):
                if self.rule_set[i].prio > r.prio:
                    cnd += 1
                    down[0] = (down[0]*(cnd-1) + self.rule_set[i].src[1] - self.rule_set[i].src[0])/cnd
                    down[1] = (down[1] * (cnd - 1) + self.rule_set[i].dst[1] - self.rule_set[i].dst[0]) / cnd
                    down[2] = (down[2] * (cnd - 1) + self.rule_set[i].src[0]) / cnd
                    down[3] = (down[3]*(cnd-1) + self.rule_set[i].dst[0])/cnd
                    down[4] = (down[4] * (cnd - 1) + self.rule_set[i].prio) / cnd
                    down[5] = cnd
                else:
                    cnu += 1
                    up[0] = (up[0]*(cnu-1) + self.rule_set[i].src[1] - self.rule_set[i].src[0])/cnu
                    up[1] = (up[1] * (cnu - 1) + self.rule_set[i].dst[1] - self.rule_set[i].dst[0]) / cnu
                    up[2] = (up[2] * (cnu - 1) + self.rule_set[i].src[0]) / cnu
                    up[3] = (up[3]*(cnu-1) + self.rule_set[i].dst[0])/cnu
                    up[4] = (up[4] * (cnu - 1) + self.rule_set[i].prio) / cnu
                    up[5] = cnu
        rself = [r.src[1]-r.src[0], r.dst[1] - r.dst[0], r.src[0], r.dst[0], r.prio]
        return up + rself + down


if __name__ == "__main__":
    tcam1 = Tcam()
    tcam2 = Tcam()

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

