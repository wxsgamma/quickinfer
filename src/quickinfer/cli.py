import argparse

def run_server(args:argparse.Namespace):
    from quickinfer.llmengine.llmegine import LLMengine
    llm=LLMengine(args)
    llm.run()
    
def run_bench(args:argparse.Namespace):
    return print(args)
def build_parser()->argparse.ArgumentParser:
    parser =argparse.ArgumentParser(
        prog="quickinfer",
        description="a quick infer"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )
    bench_parser=subparsers.add_parser(
        "benchmark",
        help="benchmark test"
    )
    serve_parser=subparsers.add_parser(
        "serve",
        help="start infer server"
    )
    serve_parser.add_argument("--model",required=True)
    serve_parser.add_argument("--port",type=int,default=8080)
    serve_parser.add_argument("--dtype",choices=["fp16","fp8","fp32","bf16"],default="fp32")
    serve_parser.set_defaults(func=run_server)
    bench_parser.set_defaults(func=run_bench)
    return parser
    
def main()->int:
    parser=build_parser()
    args=parser.parse_args()
    return args.func(args)
    
if __name__=="__main__":
    raise SystemExit(main())