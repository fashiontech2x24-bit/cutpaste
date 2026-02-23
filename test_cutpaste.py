"""Test script for cutpaste background replacement."""

from cutpaste import replace_background

if __name__ == "__main__":
    output = replace_background(
        portrait_path="test/portrait.jpg",
        background_path="test/background.png",
        output_path="test/output.png",
        prompt="person",
        confidence_threshold=0.5,
        feather_sigma=3.0,
    )
    print(f"Done! Check {output}")
