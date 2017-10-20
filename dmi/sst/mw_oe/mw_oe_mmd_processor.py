import argparse
import sys

import xarray as xr


class MwOeMMDProcessor:
    _version = "0.0.1"

    def run(self, cmd_line_args):
        input_file = cmd_line_args.input_file

        # xarray can not handle the TAI 1993 time coding @todo 3 tb/th adapt if possible
        xr.open_dataset(input_file, decode_times=False)
        


def main(args=None) -> int:
    parser = create_cmd_line_parser()
    cmd_line_args = parser.parse_args(args)

    processor = MwOeMMDProcessor()
    processor.run(cmd_line_args)

    return 0


def create_cmd_line_parser():
    parser = argparse.ArgumentParser(description='Microwave OE SST retrieval')
    parser.add_argument('input_file')
    return parser


if __name__ == "__main__":
    sys.exit(main())