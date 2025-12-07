# -*- coding: utf-8 -*-

import re
import sys
from string import punctuation
from typing import List, Tuple, Dict

# Temporary replacement
# The descriptions that contain () at the end must adapt to the new policy later
PUNCTUATION = punctuation.replace('()', '')

ANCHOR = '###'
AUTH_KEYS = ['apiKey', 'OAuth', 'X-Mashape-Key', 'User-Agent', 'No']
HTTPS_KEYS = ['Yes', 'No']
CORS_KEYS = ['Yes', 'No', 'Unknown']

INDEX_TITLE = 0
INDEX_DESC = 1
INDEX_AUTH = 2
INDEX_HTTPS = 3
INDEX_CORS = 4

NUM_SEGMENTS = 5
MIN_ENTRIES_PER_CATEGORY = 3
MAX_DESCRIPTION_LENGTH = 100

ANCHOR_RE = re.compile(ANCHOR + r'\s(.+)')
CATEGORY_TITLE_IN_INDEX_RE = re.compile(r'\*\s\[(.*)\]')
LINK_RE = re.compile(r'\[(.+)\]\((http.*)\)')

# Type aliases
APIList = List[str]
Categories = Dict[str, APIList]
CategoriesLineNumber = Dict[str, int]


def error_message(line_number: int, message: str) -> str:
    """Формирует сообщение об ошибке с номером строки."""
    line = line_number + 1
    return f'(L{line:03d}) {message}'


def get_categories_content(contents: List[str]) -> Tuple[Categories, CategoriesLineNumber]:
    """Извлекает категории и их содержимое из текста."""
    categories = {}
    category_line_num = {}

    for line_num, line_content in enumerate(contents):
        if line_content.startswith(ANCHOR):
            category = line_content.split(ANCHOR)[1].strip()
            categories[category] = []
            category_line_num[category] = line_num
            continue

        if line_content.startswith('|---') or not line_content.startswith('|'):
            continue

        raw_title = [
            raw_content.strip()
            for raw_content in line_content.split('|')[1:-1]
        ][0]

        title_match = LINK_RE.match(raw_title)
        if title_match:
            title = title_match.group(1).upper()
            categories[category].append(title)

    return categories, category_line_num


def check_alphabetical_order(lines: List[str]) -> List[str]:
    """Проверяет, что записи в каждой категории отсортированы по алфавиту."""
    error_messages = []

    categories, category_line_num = get_categories_content(contents=lines)

    for category, api_list in categories.items():
        if api_list != sorted(api_list):
            err_msg = error_message(
                category_line_num[category],
                f'Категория "{category}" не отсортирована по алфавиту'
            )
            error_messages.append(err_msg)

    return error_messages


def check_title(line_num: int, raw_title: str) -> List[str]:
    """Проверяет корректность заголовка записи."""
    error_messages = []

    title_match = LINK_RE.match(raw_title)

    if not title_match:
        err_msg = error_message(
            line_num,
            'Синтаксис заголовка должен быть "[НАЗВАНИЕ](ССЫЛКА)"'
        )
        error_messages.append(err_msg)
    elif title_match.group(1).upper().endswith(' API'):
        err_msg = error_message(
            line_num,
            'Заголовок не должен заканчиваться на "... API". '
            'Здесь каждая запись и так является API!'
        )
        error_messages.append(err_msg)

    return error_messages


def check_description(line_num: int, description: str) -> List[str]:
    """Проверяет корректность описания записи."""
    error_messages = []

    if not description:
        err_msg = error_message(line_num, 'Описание не может быть пустым')
        error_messages.append(err_msg)
        return error_messages

    first_char = description[0]
    if not first_char.isupper():
        err_msg = error_message(
            line_num,
            'Первая буква описания должна быть заглавной'
        )
        error_messages.append(err_msg)

    last_char = description[-1]
    if last_char in PUNCTUATION:
        err_msg = error_message(
            line_num,
            f'Описание не должно заканчиваться символом "{last_char}"'
        )
        error_messages.append(err_msg)

    if len(description) > MAX_DESCRIPTION_LENGTH:
        err_msg = error_message(
            line_num,
            f'Длина описания не должна превышать {MAX_DESCRIPTION_LENGTH} '
            f'символов (сейчас: {len(description)})'
        )
        error_messages.append(err_msg)

    return error_messages


def check_auth(line_num: int, auth: str) -> List[str]:
    """Проверяет корректность поля аутентификации."""
    error_messages = []

    backtick = '`'

    # Проверка обратных кавычек для не-"No" значений
    if auth != 'No' and not (auth.startswith(backtick) and auth.endswith(backtick)):
        err_msg = error_message(
            line_num,
            'Значение Auth должно быть заключено в `обратные кавычки`'
        )
        error_messages.append(err_msg)

    # Удаляем кавычки для проверки допустимых значений
    auth_value = auth.strip(backtick)
    if auth_value not in AUTH_KEYS:
        err_msg = error_message(
            line_num,
            f'"{auth}" не является допустимым значением для Auth'
        )
        error_messages.append(err_msg)

    return error_messages


def check_https(line_num: int, https: str) -> List[str]:
    """Проверяет корректность поля HTTPS."""
    error_messages = []

    if https not in HTTPS_KEYS:
        err_msg = error_message(
            line_num,
            f'"{https}" не является допустимым значением для HTTPS'
        )
        error_messages.append(err_msg)

    return error_messages


def check_cors(line_num: int, cors: str) -> List[str]:
    """Проверяет корректность поля CORS."""
    error_messages = []

    if cors not in CORS_KEYS:
        err_msg = error_message(
            line_num,
            f'"{cors}" не является допустимым значением для CORS'
        )
        error_messages.append(err_msg)

    return error_messages


def check_entry(line_num: int, segments: List[str]) -> List[str]:
    """Проверяет все поля одной записи."""
    raw_title = segments[INDEX_TITLE]
    description = segments[INDEX_DESC]
    auth = segments[INDEX_AUTH]
    https = segments[INDEX_HTTPS]
    cors = segments[INDEX_CORS]

    error_messages = []

    error_messages.extend(check_title(line_num, raw_title))
    error_messages.extend(check_description(line_num, description))
    error_messages.extend(check_auth(line_num, auth))
    error_messages.extend(check_https(line_num, https))
    error_messages.extend(check_cors(line_num, cors))

    return error_messages


def check_segment_spacing(line_num: int, segment: str) -> List[str]:
    """Проверяет отступы в сегменте строки."""
    error_messages = []

    left_spaces = len(segment) - len(segment.lstrip())
    right_spaces = len(segment) - len(segment.rstrip())

    if left_spaces != 1 or right_spaces != 1:
        err_msg = error_message(
            line_num,
            'Каждый сегмент должен начинаться и заканчиваться ровно одним пробелом'
        )
        error_messages.append(err_msg)

    return error_messages


def check_file_format(lines: List[str]) -> List[str]:
    """Проверяет формат всего файла на соответствие стандартам."""
    error_messages = []
    category_title_in_index = []

    error_messages.extend(check_alphabetical_order(lines))

    num_in_category = MIN_ENTRIES_PER_CATEGORY + 1
    category = ''
    category_line = 0

    for line_num, line_content in enumerate(lines):
        category_title_match = CATEGORY_TITLE_IN_INDEX_RE.match(line_content)
        if category_title_match:
            category_title_in_index.append(category_title_match.group(1))

        # Проверка заголовков категорий
        if line_content.startswith(ANCHOR):
            category_match = ANCHOR_RE.match(line_content)

            if category_match:
                category_name = category_match.group(1)
                if category_name not in category_title_in_index:
                    err_msg = error_message(
                        line_num,
                        f'Заголовок категории "{category_name}" '
                        f'не добавлен в раздел Index'
                    )
                    error_messages.append(err_msg)
            else:
                err_msg = error_message(
                    line_num,
                    'Заголовок категории имеет неверный формат'
                )
                error_messages.append(err_msg)

            # Проверка минимального количества записей в предыдущей категории
            if num_in_category < MIN_ENTRIES_PER_CATEGORY:
                err_msg = error_message(
                    category_line,
                    f'Категория "{category}" содержит менее '
                    f'{MIN_ENTRIES_PER_CATEGORY} записей '
                    f'(имеется: {num_in_category})'
                )
                error_messages.append(err_msg)

            category = line_content.split(' ')[1]
            category_line = line_num
            num_in_category = 0
            continue

        # Пропускаем ненужные строки
        if line_content.startswith('|---') or not line_content.startswith('|'):
            continue

        num_in_category += 1
        segments = line_content.split('|')[1:-1]

        if len(segments) < NUM_SEGMENTS:
            err_msg = error_message(
                line_num,
                f'Запись имеет не все необходимые колонки '
                f'(имеется: {len(segments)}, требуется: {NUM_SEGMENTS})'
            )
            error_messages.append(err_msg)
            continue

        # Проверка отступов в каждом сегменте
        for segment in segments:
            error_messages.extend(check_segment_spacing(line_num, segment))

        segments = [segment.strip() for segment in segments]
        error_messages.extend(check_entry(line_num, segments))

    return error_messages


def main(filename: str) -> None:
    """Основная функция для проверки файла."""
    try:
        with open(filename, mode='r', encoding='utf-8') as file:
            lines = [line.rstrip() for line in file]
    except FileNotFoundError:
        print(f'Ошибка: файл "{filename}" не найден')
        sys.exit(1)
    except UnicodeDecodeError:
        print(f'Ошибка: файл "{filename}" имеет неверную кодировку')
        sys.exit(1)

    file_format_err_msgs = check_file_format(lines)

    if file_format_err_msgs:
        for err_msg in file_format_err_msgs:
            print(err_msg)
        sys.exit(1)

    print(f'Файл "{filename}" успешно прошел проверку формата')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Ошибка: не указан файл .md для проверки')
        print('Использование: python format.py <filename.md>')
        sys.exit(1)

    filename = sys.argv[1]

    if not filename.endswith('.md'):
        print('Предупреждение: рекомендуется использовать файлы с расширением .md')

    main(filename)