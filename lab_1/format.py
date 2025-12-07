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
    line = line_number + 1
    return f'(L{line:03d}) {message}'


def get_categories_content(contents: List[str]) -> Tuple[Categories, CategoriesLineNumber]:
    categories = {}
    category_line_num = {}

    for line_num, line_content in enumerate(contents):
        if line_content.startswith(ANCHOR):
            category = line_content.split(ANCHOR)[1].strip()
            categories[category] = []
            category_line_num[category] = line_num
            continue

        if not line_content.startswith('|') or line_content.startswith('|---'):
            continue

        raw_title = [
            raw_content.strip() for raw_content in line_content.split('|')[1:-1]
        ][0]

        title_match = LINK_RE.match(raw_title)
        if title_match:
            title = title_match.group(1).upper()
            categories[category].append(title)

    return categories, category_line_num


def check_alphabetical_order(lines: List[str]) -> List[str]:
    error_messages = []

    categories, category_line_num = get_categories_content(contents=lines)

    for category, api_list in categories.items():
        if sorted(api_list) != api_list:
            err_msg = error_message(
                category_line_num[category],
                f'{category} category is not alphabetical order'
            )
            error_messages.append(err_msg)

    return error_messages


def check_title(line_num: int, raw_title: str) -> List[str]:
    error_messages = []

    title_match = LINK_RE.match(raw_title)

    # url should be wrapped in "[TITLE](LINK)" Markdown syntax
    if not title_match:
        err_msg = error_message(line_num, 'Title syntax should be "[TITLE](LINK)"')
        error_messages.append(err_msg)
    else:
        # do not allow "... API" in the entry title
        title = title_match.group(1)
        if title.upper().endswith(' API'):
            err_msg = error_message(
                line_num,
                'Title should not end with "... API". Every entry is an API here!'
            )
            error_messages.append(err_msg)

    return error_messages


def check_description(line_num: int, description: str) -> List[str]:
    error_messages = []

    first_char = description[0]
    if first_char.upper() != first_char:
        err_msg = error_message(
            line_num,
            'first character of description is not capitalized'
        )
        error_messages.append(err_msg)

    last_char = description[-1]
    if last_char in PUNCTUATION:
        err_msg = error_message(
            line_num,
            f'description should not end with {last_char}'
        )
        error_messages.append(err_msg)

    desc_length = len(description)
    if desc_length > MAX_DESCRIPTION_LENGTH:
        err_msg = error_message(
            line_num,
            f'description should not exceed {MAX_DESCRIPTION_LENGTH} '
            f'characters (currently {desc_length})'
        )
        error_messages.append(err_msg)

    return error_messages


def check_auth(line_num: int, auth: str) -> List[str]:
    error_messages = []

    backtick = '`'
    if auth != 'No' and (not auth.startswith(backtick) or not auth.endswith(backtick)):
        err_msg = error_message(
            line_num,
            'auth value is not enclosed with `backticks`'
        )
        error_messages.append(err_msg)

    if auth.replace(backtick, '') not in AUTH_KEYS:
        err_msg = error_message(
            line_num,
            f'{auth} is not a valid Auth option'
        )
        error_messages.append(err_msg)

    return error_messages


def check_https(line_num: int, https: str) -> List[str]:
    error_messages = []

    if https not in HTTPS_KEYS:
        err_msg = error_message(
            line_num,
            f'{https} is not a valid HTTPS option'
        )
        error_messages.append(err_msg)

    return error_messages


def check_cors(line_num: int, cors: str) -> List[str]:
    error_messages = []

    if cors not in CORS_KEYS:
        err_msg = error_message(
            line_num,
            f'{cors} is not a valid CORS option'
        )
        error_messages.append(err_msg)

    return error_messages


def check_entry(line_num: int, segments: List[str]) -> List[str]:
    raw_title = segments[INDEX_TITLE]
    description = segments[INDEX_DESC]
    auth = segments[INDEX_AUTH]
    https = segments[INDEX_HTTPS]
    cors = segments[INDEX_CORS]

    title_err_msgs = check_title(line_num, raw_title)
    desc_err_msgs = check_description(line_num, description)
    auth_err_msgs = check_auth(line_num, auth)
    https_err_msgs = check_https(line_num, https)
    cors_err_msgs = check_cors(line_num, cors)

    error_messages = [
        *title_err_msgs,
        *desc_err_msgs,
        *auth_err_msgs,
        *https_err_msgs,
        *cors_err_msgs
    ]

    return error_messages


def check_file_format(lines: List[str]) -> List[str]:
    error_messages = []
    category_title_in_index = []

    alphabetical_err_msgs = check_alphabetical_order(lines)
    error_messages.extend(alphabetical_err_msgs)

    num_in_category = MIN_ENTRIES_PER_CATEGORY + 1
    category = ''
    category_line = 0

    for line_num, line_content in enumerate(lines):
        category_title_match = CATEGORY_TITLE_IN_INDEX_RE.match(line_content)
        if category_title_match:
            category_title_in_index.append(category_title_match.group(1))

        # check each category for the minimum number of entries
        if line_content.startswith(ANCHOR):
            category_match = ANCHOR_RE.match(line_content)
            if category_match:
                if category_match.group(1) not in category_title_in_index:
                    err_msg = error_message(
                        line_num,
                        f'category header ({category_match.group(1)}) '
                        f'not added to Index section'
                    )
                    error_messages.append(err_msg)
            else:
                err_msg = error_message(
                    line_num,
                    'category header is not formatted correctly'
                )
                error_messages.append(err_msg)

            if num_in_category < MIN_ENTRIES_PER_CATEGORY:
                err_msg = error_message(
                    category_line,
                    f'{category} category does not have the minimum '
                    f'{MIN_ENTRIES_PER_CATEGORY} entries '
                    f'(only has {num_in_category})'
                )
                error_messages.append(err_msg)

            category = line_content.split(' ')[1]
            category_line = line_num
            num_in_category = 0
            continue

        # skips lines that we do not care about
        if not line_content.startswith('|') or line_content.startswith('|---'):
            continue

        num_in_category += 1
        segments = line_content.split('|')[1:-1]
        if len(segments) < NUM_SEGMENTS:
            err_msg = error_message(
                line_num,
                f'entry does not have all the required columns '
                f'(have {len(segments)}, need {NUM_SEGMENTS})'
            )
            error_messages.append(err_msg)
            continue

        for segment in segments:
            # every line segment should start and end with exactly 1 space
            if (len(segment) - len(segment.lstrip()) != 1 or
                    len(segment) - len(segment.rstrip()) != 1):
                err_msg = error_message(
                    line_num,
                    'each segment must start and end with exactly 1 space'
                )
                error_messages.append(err_msg)

        segments = [segment.strip() for segment in segments]
        entry_err_msgs = check_entry(line_num, segments)
        error_messages.extend(entry_err_msgs)

    return error_messages


def main(filename: str) -> None:
    with open(filename, mode='r', encoding='utf-8') as file:
        lines = list(line.rstrip() for line in file)

    file_format_err_msgs = check_file_format(lines)

    if file_format_err_msgs:
        for err_msg in file_format_err_msgs:
            print(err_msg)
        sys.exit(1)


if __name__ == '__main__':
    num_args = len(sys.argv)

    if num_args < 2:
        print('No .md file passed (file should contain Markdown table syntax)')
        sys.exit(1)

    filename = sys.argv[1]

    main(filename)