"""M0 acceptance check: verify ffmpeg, GPU/CUDA, and Ollama are all working.

Run with: python scripts/check_env.py
"""
import shutil
import subprocess
import sys
import urllib.request
import urllib.error

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

OLLAMA_HOST = "http://localhost:11434"


def ok(msg: str) -> None:
    print(f"{GREEN}[OK]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}[FAIL]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def check_ffmpeg() -> bool:
    path = shutil.which("ffmpeg")
    if not path:
        fail("ffmpeg not found on PATH")
        return False
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        ).stdout
        version_line = out.splitlines()[0]
        ok(f"ffmpeg found at {path} ({version_line})")
        return True
    except Exception as e:
        fail(f"ffmpeg found at {path} but failed to run: {e}")
        return False


def check_gpu() -> bool:
    nvidia_smi_ok = False
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            ok(f"nvidia-smi reports GPU: {out}")
            nvidia_smi_ok = True
        except Exception as e:
            fail(f"nvidia-smi present but failed to run: {e}")
    else:
        fail("nvidia-smi not found on PATH (no NVIDIA driver visible in this environment)")

    ct2_ok = False
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        if count > 0:
            ok(f"ctranslate2 sees {count} CUDA device(s)")
            ct2_ok = True
        else:
            fail("ctranslate2 is installed but reports 0 CUDA devices")
    except ImportError:
        warn("ctranslate2 not installed yet (install requirements.txt) — skipping CTranslate2 CUDA check")
        ct2_ok = None
    except Exception as e:
        fail(f"ctranslate2 CUDA check errored: {e}")

    if ct2_ok is None:
        return nvidia_smi_ok
    return nvidia_smi_ok and ct2_ok


def check_ollama() -> bool:
    path = shutil.which("ollama")
    if not path:
        fail("ollama binary not found on PATH")
        return False
    ok(f"ollama binary found at {path}")

    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            import json

            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            ok(f"Ollama server reachable at {OLLAMA_HOST}")
            if models:
                ok(f"Installed models: {', '.join(models)}")
            else:
                warn(
                    "Ollama server is running but no models are pulled yet. "
                    "Run: ollama pull qwen2.5:3b-instruct"
                )
            return True
    except urllib.error.URLError as e:
        fail(
            f"Could not reach Ollama server at {OLLAMA_HOST} ({e}). "
            "Is it running? Try: ollama serve"
        )
        return False
    except Exception as e:
        fail(f"Unexpected error contacting Ollama: {e}")
        return False


def main() -> int:
    print("=== Video Understanding — environment check (M0) ===\n")

    print("-- ffmpeg --")
    ffmpeg_ok = check_ffmpeg()

    print("\n-- GPU / CUDA --")
    gpu_ok = check_gpu()

    print("\n-- Ollama --")
    ollama_ok = check_ollama()

    print("\n=== Summary ===")
    results = {"ffmpeg": ffmpeg_ok, "GPU/CUDA": gpu_ok, "Ollama": ollama_ok}
    for name, passed in results.items():
        (ok if passed else fail)(name)

    all_ok = all(results.values())
    if all_ok:
        print(f"\n{GREEN}All checks passed — ready for M1.{RESET}")
    else:
        print(f"\n{RED}Some checks failed — resolve above before moving on.{RESET}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
