import argparse
import sys

from dmi.sst.mw_oe.mw_oe_sst_processor import MwOeSstProcessor


def main(args=None) -> int:
    parser = create_cmd_line_parser()
    cmd_line_args = parser.parse_args(args)

    processor = MwOeSstProcessor()
    processor.run(cmd_line_args)

    return 0


def create_cmd_line_parser():
    parser = argparse.ArgumentParser(description='Microwave OE SST retrieval')
    parser.add_argument('input_file')
    return parser


if __name__ == "__main__":
    sys.exit(main())
