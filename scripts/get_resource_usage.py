#!/usr/bin/env python

import re
import sys

def match(regex, line):
    match_object = re.match(regex, line.strip())
    return (match_object.group(1), match_object.group(2))


def main(argv):
    filename = sys.argv[1]
    state = -1
    arr = []
    regexes = [
            r".*Logic utilization:\s+(\d+)\s+/\s+(\d+)\s+.*$",
            r".*Primary FFs:\s+(\d+)\s+/\s+(\d+)\s+.*$",
            r".*Secondary FFs:\s+(\d+)\s+/\s+(\d+)\s+.*$",
            r".*Multipliers \(18x18\):\s+(\d+)\s+/\s+(\d+)\s+.*$",
            r".*DSP blocks:\s+(\d+)\s+/\s+(\d+)\s+.*$",
            r".*Block memory \(M20K\):\s+(\d+)\s+/\s+(\d+)\s+.*$"]

    with open(filename, "r") as f:
        for line in f:
            if state == -1:
                if re.match(
                        r".*FINAL RESOURCE USAGE$",
                        line.strip()):
                    state += 1
            elif state >= 0 and state <= 5:
                arr.append(match(regexes[state],
                                 line.strip()))
                state += 1

    x = []
    for a, b in arr:
        x.append(a)
        x.append(b)

    print ",".join(x)
        


if __name__ == "__main__":
    main(sys.argv)
