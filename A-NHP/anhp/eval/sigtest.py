# -*- coding: utf-8 -*-
# !/usr/bin/python
import numpy

class Bootstrapping(object):

    def __init__(self):
        self.eps = numpy.finfo(float).eps

    def run(self, input, num=1000):
        """
        input is a list of tuples (weighted score, weight)
        weight is unnormalized
        score has been weighted
        e.g. if it is the log lik of a seq
        then weighted score is the log lik of the entire seq
        weight is the len of the seq
        """
        numerator = []
        denominator = []
        for nume, deno in input:
            numerator.append(nume)
            denominator.append(deno)
        numerator = numpy.array(numerator)
        denominator = numpy.array(denominator)

        le = len(input)
        res = []
        for i in range(num):
            selected = numpy.random.choice(a=le, size=le, replace=True)
            sample = numpy.sum(numerator[selected]) / numpy.sum(denominator[selected])
            res.append( sample )
        res = numpy.array(sorted(res))

        print(f"bootstrapping {num} times")
        avg = numpy.mean(res)
        print(f"mean is {avg:.4f}")
        for cl in [95, 99]:
            low_q = 0.5 * (100.0 - cl)
            high_q = low_q + cl
            low = numpy.percentile(a=res, q=low_q)
            high = numpy.percentile(a=res, q=high_q)
            print(f"for confidence level {cl}%, confidence interval is {low:.4f} to {high:.4f}")


class PairPerm(object):

    def __init__(self):
        self.eps = numpy.finfo(float).eps

    def run(self, input1, input2, num=10000):
        """
        input1 and input2 : list of tuples (weighted score, weight)
        weight is unnormalized
        similar to input of Bootstrapping
        """
        x, y = [], []
        for nume_1_deno_1, nume_2_deno_2 in zip(input1, input2):
            x.append(nume_1_deno_1[0])
            y.append(nume_2_deno_2[0])
        x, y = map(numpy.array, [x, y])
        d = x - y
        le = len(d)
        mu = numpy.sum(d)
        perm = (numpy.random.binomial(1, .5, (num, le)) * 2 - 1) * d
        mu_perm = numpy.sum(perm, 1)

        print(f"{le} data points, {100.0*sum(d==0)/le:.2f}% equal")
        print(f"not equal: {100.0*sum(d>0)/le:.2f}% vs. {100.0*sum(d<0)/le:.2f}%")

        pvalue = float(sum(abs(mu_perm) >= abs(mu))) / num
        print(f"paired permutation test with {num} random samples")
        print(f"p value is {pvalue:.4f}")
        return pvalue
