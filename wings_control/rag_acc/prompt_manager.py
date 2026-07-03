import re
import os
import time
import logging
import weakref
import threading
from functools import wraps

_template_manager_lock = threading.Lock()
template_manager = None


def singleton(cls):
    instances = weakref.WeakValueDictionary()
    lock = threading.Lock()

    @wraps(cls)
    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                instance = cls(*args, **kwargs)
                instances[cls] = instance
            return instances[cls]

    return get_instance


@singleton
class TemplateManager:
    def __init__(self, reload_on_get=False, template_dir="templates"):
        self.reload_on_get = reload_on_get
        self.template_dir = os.path.normpath(template_dir)
        self.templates = {"zh": None, "en": None}
        self.last_loaded_time = {"zh": None, "en": None}
        if not self.reload_on_get:
            self._load_templates("zh")
            self._load_templates("en")

    def get_template(self, lang, prompt_type):
        if self.reload_on_get:
            self._load_templates(lang)
        return self.templates.get(lang, {}).get(prompt_type)

    def _load_templates(self, lang):
        current_dir = os.path.dirname(__file__)
        real_current_dir = os.path.realpath(current_dir)
        full_template_dir = os.path.normpath(
            os.path.join(real_current_dir, self.template_dir))
        real_full_template_dir = os.path.realpath(full_template_dir)

        if not real_full_template_dir.startswith(real_current_dir):
            raise ValueError(f"Invalid template directory path: {real_full_template_dir}")

        lang_dir = os.path.join(real_full_template_dir, lang)
        normalized_lang_dir = os.path.normpath(lang_dir)
        real_lang_dir = os.path.realpath(normalized_lang_dir)

        if not real_lang_dir.startswith(real_full_template_dir):
            raise ValueError(f"Language directory path traversal detected: {lang_dir}")

        if not os.path.exists(real_lang_dir):
            raise FileNotFoundError(f"Template directory does not exist: {real_lang_dir}")

        templates = {}
        for key in ["map", "combine"]:
            file_path = os.path.join(real_lang_dir, f"{key}.txt")
            normalized_file_path = os.path.normpath(file_path)
            real_file_path = os.path.realpath(normalized_file_path)

            if not real_file_path.startswith(real_full_template_dir):
                raise ValueError(f"Invalid template file path: {real_file_path}")

            if not os.path.exists(real_file_path):
                raise FileNotFoundError(f"Template file does not exist: {real_file_path}")

            try:
                with open(real_file_path, "r", encoding="utf-8") as f:
                    templates[key] = f.read()
            except PermissionError as e:
                logging.error("Permission denied while reading template file %s: %s", real_file_path, e)
                raise
            except IOError as e:
                logging.error("IO error while reading template file %s: %s", real_file_path, e)
                raise
            except Exception as e:
                logging.error("Unexpected error while reading template file %s: %s", real_file_path, e)
                raise

        self.templates[lang] = templates
        self.last_loaded_time[lang] = time.time()


def init_template_manager(reload_on_get=False):
    global template_manager
    if template_manager is None:
        with _template_manager_lock:
            if template_manager is None:
                template_manager = TemplateManager(reload_on_get=reload_on_get)
    return template_manager


def is_chinese_text(text, threshold=0.05):
    if not text or len(text) == 0:
        return False
    if not isinstance(text, str):
        return True
    chinese_chars = re.findall(r"[\u4E00-\u9FFF]", text)
    return len(chinese_chars) / len(text) >= threshold


def generate_prompt(prompt_type, question, context, analysis=None):
    lang = "zh" if is_chinese_text(question) else "en"
    manager = init_template_manager()
    template = manager.get_template(lang, prompt_type)
    if not template:
        raise ValueError(f"Unsupported prompt type or language: {prompt_type}, {lang}")
    try:
        if analysis:
            return template.format(context=context, question=question, analysis=analysis)
        return template.format(context=context, question=question)
    except (KeyError, IndexError) as e:
        logging.error(
            f"Template formatting error for {prompt_type} in {lang} with placeholders: context, question. "
            f"Error type: {type(e).__name__}")
        raise ValueError(f"Template formatting failed: missing placeholder in template") from e


def generate_map_prompt(question, context):
    return generate_prompt("map", question, context)


def generate_combine_prompt(question, context, preliminary_analysis=None):
    return generate_prompt("combine", question, context, preliminary_analysis)
