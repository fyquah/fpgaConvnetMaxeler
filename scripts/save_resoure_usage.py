#!/usr/bin/env python

import argparse
import collections
import datetime
import os
import re
import sys

import yaml


parser = argparse.ArgumentParser(
        description="Parses the build logs to get resource usages information"
                    " and write them to a yaml file.")
parser.add_argument("logs", type=str, nargs="+",
                    help="list of build logs to parse. The model name will be"
                         " identified as <model_name>/_build.log.")
parser.add_argument("--output", dest="output", type=str,
                    help=("yaml file to store the output of parsing. If there"
                          " are existing files in the yaml file, the script"
                          " appends the yaml file rather than modifying it."),
                    required=True)
parser.add_argument("--commit-hash", dest="commit_hash", type=str,
                    help="The commit has that the results belongs to.",
                    required=True)
                     


# ref: https://gist.github.com/miracle2k/3184458
def represent_odict(dump, tag, mapping, flow_style=None):
    """Like BaseRepresenter.represent_mapping, but does not issue the sort().
    """
    value = []
    node = yaml.MappingNode(tag, value, flow_style=flow_style)
    if dump.alias_key is not None:
        dump.represented_objects[dump.alias_key] = node
    best_style = True
    if hasattr(mapping, 'items'):
        mapping = mapping.items()
    for item_key, item_value in mapping:
        node_key = dump.represent_data(item_key)
        node_value = dump.represent_data(item_value)
        if not (isinstance(node_key, yaml.ScalarNode) and not node_key.style):
            best_style = False
        if not (isinstance(node_value, yaml.ScalarNode) and not node_value.style):
            best_style = False
        value.append((node_key, node_value))
    if flow_style is None:
        if dump.default_flow_style is not None:
            node.flow_style = dump.default_flow_style
        else:
            node.flow_style = best_style
    return node


yaml.SafeDumper.add_representer(collections.OrderedDict,
    lambda dumper, value: represent_odict(dumper, u'tag:yaml.org,2002:map', value))

# For loading yaml as ordereddict.
# ref: http://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

def dict_representer(dumper, data):
    return dumper.represent_dict(data.iteritems())

def dict_constructor(loader, node):
    return collections.OrderedDict(loader.construct_pairs(node))

yaml.add_representer(collections.OrderedDict, dict_representer)
yaml.add_constructor(_mapping_tag, dict_constructor)


def match(regex, line):
    match_object = re.match(regex, line.strip())
    if match_object:
        return (match_object.group(1), match_object.group(2))
    else:
        return None


def parse_file(filename):
    regexes = collections.OrderedDict([
            ("logic_utilization", r".*Logic utilization:\s+(\d+)\s+/\s+(\d+)\s+.*$"),
            ("primary_ff", r".*Primary FFs:\s+(\d+)\s+/\s+(\d+)\s+.*$"),
            ("seconday_ff", r".*Secondary FFs:\s+(\d+)\s+/\s+(\d+)\s+.*$"),
            ("multiplier", ".*Multipliers \(18x18\):\s+(\d+)\s+/\s+(\d+)\s+.*$"),
            ("dsp_block", r".*DSP blocks:\s+(\d+)\s+/\s+(\d+)\s+.*$"),
            ("block_memory", r".*Block memory \(M20K\):\s+(\d+)\s+/\s+(\d+)\s+.*$")
            ])
    state = -1
    res = collections.OrderedDict()

    with open(filename, "r") as f:
        for line in f:
            if state == -1:
                if re.match(
                        r".*FINAL RESOURCE USAGE$",
                        line.strip()):
                    state += 1
            elif state >= 0 and state < 6:
                for name, regex in regexes.iteritems():
                    match_object = match(regex, line.strip())
                    if match_object:
                        state += 1
                        res[name] = int(match_object[0])
                        break
    return dict([
        ("name", os.path.basename(os.path.dirname(filename))),
        ("time", str(datetime.datetime.fromtimestamp(os.stat(filename).st_mtime))),
        ("usage", res)])


def is_duplicate(records, new_entry):
    for record in records:
        if (record["name"] == new_entry["name"]
                and record["time"] == new_entry["time"]
                and record["commit_hash"] == new_entry["commit_hash"]):
            return True


def main(argv):
    results = collections.OrderedDict([
            ("resources", collections.OrderedDict([
                ("logic_utilization", 262400),
                ("primary_ff", 524800),
                ("secondary_ff", 524800),
                ("multiplier", 3926),
                ("dsp_block", 1963),
                ("block_memory", 2567),
            ])),
            ("data", [])])

    if os.path.exists(FLAGS.output):
        with open(FLAGS.output, "r") as f:
            results["data"] = yaml.load(f.read())["data"]

    for filename in FLAGS.logs:
        new_result = parse_file(filename)
        new_result["commit_hash"] = FLAGS.commit_hash

        if new_result and not is_duplicate(results["data"], new_result):
            results["data"].append(new_result)

    with open(FLAGS.output, "w") as f:
        f.write(yaml.dump(results, default_flow_style=False))

    print yaml.dump(results, default_flow_style=False)


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    main(sys.argv)
