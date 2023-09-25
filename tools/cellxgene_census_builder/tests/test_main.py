from cellxgene_census_builder.build_soma.__main__ import create_args_parser


def test_create_args_parser_default_build() -> None:
    parser = create_args_parser()
    args_list = [".", "build"]

    args = parser.parse_args(args=args_list)

    assert args.working_dir == "."
    assert args.verbose == 0
    assert args.multi_process is False
    assert isinstance(args.build_tag, str)
    assert args.subcommand == "build"
    assert args.validate is True
    assert args.consolidate is True


def test_create_args_parser_default_validate() -> None:
    parser = create_args_parser()
    args_list = [".", "validate"]

    args = parser.parse_args(args=args_list)

    assert args.working_dir == "."
    assert args.verbose == 0
    assert args.multi_process is False
    assert isinstance(args.build_tag, str)
    assert args.subcommand == "validate"
