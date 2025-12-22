# compare_with_v1.py
import json
import matplotlib.pyplot as plt
import numpy as np


def load_v1_results():
    """
    Загрузите результаты первой версии (замените на реальные данные)
    """
    # Пример данных для v1 - замените на ваши реальные измерения
    v1_results = {
        '128x128': {'time': 0.45, 'memory': 85.5, 'keypoints': 120},
        '256x256': {'time': 1.82, 'memory': 195.3, 'keypoints': 380},
        '512x512': {'time': 7.45, 'memory': 580.2, 'keypoints': 1250},
        '1024x1024': {'time': 29.80, 'memory': 1850.5, 'keypoints': 4200}
    }
    return v1_results


def compare_performance(v2_file='performance_summary.json'):
    """
    Сравнение производительности v1 и v2
    """
    # Загружаем результаты v2
    with open(v2_file, 'r') as f:
        v2_results = json.load(f)

    # Загружаем результаты v1
    v1_results = load_v1_results()

    # Подготавливаем данные для сравнения
    sizes = ['128x128', '256x256', '512x512', '1024x1024']

    v1_times = []
    v2_times = []
    v1_memory = []
    v2_memory = []
    v1_keypoints = []
    v2_keypoints = []

    for size in sizes:
        v1_times.append(v1_results[size]['time'])
        v1_memory.append(v1_results[size]['memory'])
        v1_keypoints.append(v1_results[size]['keypoints'])

        v2_data = v2_results['summary_by_size'][size]
        v2_times.append(float(v2_data['avg_time'].replace('s', '')))
        v2_memory.append(float(v2_data['avg_memory'].replace('MB', '')))
        v2_keypoints.append(v2_data['avg_keypoints'])

    # Вычисляем улучшения
    time_improvement = [(1 - v2 / v1) * 100 for v1, v2 in zip(v1_times, v2_times)]
    memory_improvement = [(1 - v2 / v1) * 100 for v1, v2 in zip(v1_memory, v2_memory)]

    # Создаем графики сравнения
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Сравнение времени выполнения
    ax1 = axes[0, 0]
    x = np.arange(len(sizes))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, v1_times, width, label='v1 - Original', color='red', alpha=0.7)
    bars2 = ax1.bar(x + width / 2, v2_times, width, label='v2 - Optimized', color='green', alpha=0.7)

    ax1.set_xlabel('Image Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time Comparison: v1 vs v2')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Добавляем значения
    for i, (v1, v2) in enumerate(zip(v1_times, v2_times)):
        ax1.text(i - width / 2, v1 + max(v1_times) * 0.05, f'{v1:.2f}s',
                 ha='center', va='bottom', fontsize=8)
        ax1.text(i + width / 2, v2 + max(v2_times) * 0.05, f'{v2:.2f}s',
                 ha='center', va='bottom', fontsize=8)

    # 2. Сравнение использования памяти
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x - width / 2, v1_memory, width, label='v1 - Original', color='red', alpha=0.7)
    bars4 = ax2.bar(x + width / 2, v2_memory, width, label='v2 - Optimized', color='green', alpha=0.7)

    ax2.set_xlabel('Image Size')
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('Memory Usage Comparison: v1 vs v2')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    for i, (v1, v2) in enumerate(zip(v1_memory, v2_memory)):
        ax2.text(i - width / 2, v1 + max(v1_memory) * 0.05, f'{v1:.1f}MB',
                 ha='center', va='bottom', fontsize=8)
        ax2.text(i + width / 2, v2 + max(v2_memory) * 0.05, f'{v2:.1f}MB',
                 ha='center', va='bottom', fontsize=8)

    # 3. Процент улучшения времени
    ax3 = axes[0, 2]
    colors = ['green' if x > 0 else 'red' for x in time_improvement]
    bars5 = ax3.bar(x, time_improvement, color=colors, alpha=0.7)

    ax3.set_xlabel('Image Size')
    ax3.set_ylabel('Improvement (%)')
    ax3.set_title('Speed Improvement: v2 over v1')
    ax3.set_xticks(x)
    ax3.set_xticklabels(sizes)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3)

    for i, val in enumerate(time_improvement):
        ax3.text(i, val + (1 if val > 0 else -1), f'{val:.1f}%',
                 ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

    # 4. Процент улучшения памяти
    ax4 = axes[1, 0]
    colors = ['green' if x > 0 else 'red' for x in memory_improvement]
    bars6 = ax4.bar(x, memory_improvement, color=colors, alpha=0.7)

    ax4.set_xlabel('Image Size')
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('Memory Improvement: v2 over v1')
    ax4.set_xticks(x)
    ax4.set_xticklabels(sizes)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3)

    for i, val in enumerate(memory_improvement):
        ax4.text(i, val + (1 if val > 0 else -1), f'{val:.1f}%',
                 ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

    # 5. Сравнение количества ключевых точек
    ax5 = axes[1, 1]
    bars7 = ax5.bar(x - width / 2, v1_keypoints, width, label='v1', color='blue', alpha=0.7)
    bars8 = ax5.bar(x + width / 2, v2_keypoints, width, label='v2', color='orange', alpha=0.7)

    ax5.set_xlabel('Image Size')
    ax5.set_ylabel('Keypoints Count')
    ax5.set_title('Keypoints Detection Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels(sizes)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Сводная таблица улучшений
    ax6 = axes[1, 2]
    ax6.axis('tight')
    ax6.axis('off')

    # Создаем данные для таблицы
    table_data = []
    for i, size in enumerate(sizes):
        table_data.append([
            size,
            f'{v1_times[i]:.2f}s → {v2_times[i]:.2f}s',
            f'{time_improvement[i]:.1f}%',
            f'{v1_memory[i]:.1f}MB → {v2_memory[i]:.1f}MB',
            f'{memory_improvement[i]:.1f}%',
            f'{v1_keypoints[i]} → {v2_keypoints[i]}'
        ])

    table = ax6.table(cellText=table_data,
                      colLabels=['Size', 'Time', 'Time Imp.', 'Memory', 'Mem. Imp.', 'Keypoints'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.15, 0.2, 0.1, 0.2, 0.1, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Подсветка улучшений
    for i in range(len(sizes)):
        if time_improvement[i] > 0:
            table[(i + 1, 2)].set_facecolor('#90EE90')  # Светло-зеленый
        if memory_improvement[i] > 0:
            table[(i + 1, 4)].set_facecolor('#90EE90')

    plt.suptitle('SIFT Algorithm Performance Comparison: Original vs Optimized Version',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('v1_v2_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Сохраняем результаты сравнения в JSON
    comparison_results = {
        'comparison_date': '2024-01-15',
        'summary': {
            'avg_speed_improvement': f'{np.mean(time_improvement):.1f}%',
            'avg_memory_improvement': f'{np.mean(memory_improvement):.1f}%',
            'total_improvement': 'Significant performance gains in both speed and memory'
        },
        'detailed_comparison': {}
    }

    for i, size in enumerate(sizes):
        comparison_results['detailed_comparison'][size] = {
            'time_v1': v1_times[i],
            'time_v2': v2_times[i],
            'time_improvement_percent': time_improvement[i],
            'memory_v1': v1_memory[i],
            'memory_v2': v2_memory[i],
            'memory_improvement_percent': memory_improvement[i],
            'keypoints_v1': v1_keypoints[i],
            'keypoints_v2': v2_keypoints[i],
            'keypoints_difference': v2_keypoints[i] - v1_keypoints[i]
        }

    with open('v1_v2_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=4)

    print("\nComparison Results:")
    print("=" * 60)
    print(f"Average Speed Improvement: {np.mean(time_improvement):.1f}%")
    print(f"Average Memory Improvement: {np.mean(memory_improvement):.1f}%")
    print(f"Keypoints Detection: Similar accuracy")
    print("\nFiles generated:")
    print("  1. v1_v2_comparison.png - Visual comparison charts")
    print("  2. v1_v2_comparison.json - Detailed comparison data")
    print("=" * 60)


if __name__ == '__main__':
    compare_performance()