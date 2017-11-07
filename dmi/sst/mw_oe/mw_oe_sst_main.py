import sys

from dmi.sst.mw_oe.mw_oe_sst_processor import MwOeSstProcessor


def main(args=None) -> int:
    processor = MwOeSstProcessor()
    processor.run(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
