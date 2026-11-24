import base64
import io
import json
import os
import re
import time

import requests

URL = "https://api.openai.com/v1/responses"


def png64(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def safe(error):
    if isinstance(error, str) and error.startswith("HTTP "):
        match = re.match(r"HTTP \d{3}", error)
        return match.group(0) if match else "HTTP error"
    return error


class GPT:
    def __init__(self, model, effort, log=None):
        self.model, self.effort, self.log = model, effort, log
        self.calls, self.seconds = 0, 0.0
        self.key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SIS_GPT_CRED")
        if not self.key:
            raise SystemExit("OPENAI_API_KEY is not set")

    def request(self, prompt, images):
        content = [{"type": "input_text", "text": prompt}] + [
            {"type": "input_image", "image_url": "data:image/png;base64," + png64(im)} for im in images]
        body = {"model": self.model, "input": [{"role": "user", "content": content}]}
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        try:
            r = requests.post(URL, json=body, timeout=180, headers={"Authorization": "Bearer " + self.key})
            if r.status_code != 200:
                return "", "HTTP %d: %s" % (r.status_code, r.text[:200])
            j = r.json()
            raw = j.get("output_text") or "".join(
                c.get("text", "") for o in j.get("output", []) for c in (o.get("content") or [])
                if c.get("type") == "output_text")
            return raw, None
        except requests.RequestException as e:
            return "", type(e).__name__

    def ask(self, tag, text, images, check=None, tries=3):
        last_err, raw = None, ""
        prompt = text
        for attempt in range(tries):
            t0 = time.time()
            raw, err = self.request(prompt, images)
            dt = time.time() - t0
            self.calls += 1
            self.seconds += dt
            parsed = None
            if err is None:
                m = re.search(r"\{.*\}", raw, re.S)
                try:
                    parsed = json.loads(m.group(0)) if m else None
                except ValueError:
                    parsed = None
                err = None if isinstance(parsed, dict) else "no JSON object in the answer"
                if err is None and check is not None:
                    err = check(parsed)
            if self.log:
                self.log({"gpt": tag, "attempt": attempt + 1, "seconds": round(dt, 2), "error": safe(err), "raw": raw[:2000]})
            if err is None:
                return parsed, raw, None
            last_err = err
            if raw and not err.startswith("HTTP"):
                prompt = text + "\n\nYour previous answer could not be used: " + err + ". Answer again in the required format."
            else:
                time.sleep(3)
        return None, raw, safe(last_err)
