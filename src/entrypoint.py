#!/usr/bin/env python3
"""Docker entrypoint for OCR Pipeline."""
import os
import sys
import glob
from pathlib import Path


def main():
    # Find input PDF
    input_pdf = os.environ.get('OCR_INPUT_PDF')
    if not input_pdf:
        pdfs = glob.glob('/data/input/*.pdf')
        if len(pdfs) == 1:
            input_pdf = pdfs[0]
        elif len(pdfs) > 1:
            print("ERROR: Multiple PDFs in /data/input. Set OCR_INPUT_PDF env var.")
            sys.exit(1)
        else:
            print("ERROR: No PDF found in /data/input")
            sys.exit(1)

    # Output path
    input_name = Path(input_pdf).stem
    output_md = os.environ.get('OCR_OUTPUT_PATH', f'/data/output/{input_name}.md')

    # Build args
    args = [input_pdf, output_md]
    args.extend(['--classifier', os.environ.get('OCR_CLASSIFIER', 'qwen3-vl-8b')])
    args.extend(['--precision', os.environ.get('OCR_PRECISION', 'fp16')])

    if os.environ.get('OCR_DESCRIBE_DIAGRAMS', 'false').lower() == 'true':
        args.append('--describe-diagrams')

    args.extend(['--dpi', os.environ.get('OCR_DPI', '200')])
    args.append('--verbose')

    print(f"OCR Pipeline starting...")
    print(f"  Input:  {input_pdf}")
    print(f"  Output: {output_md}")
    print(f"  Args:   {' '.join(args)}")

    sys.argv = ['ocr_pipeline.py'] + args
    from ocr_pipeline import main as run_pipeline
    run_pipeline()


if __name__ == '__main__':
    main()
