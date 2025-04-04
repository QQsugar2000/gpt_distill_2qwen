prompt = '''帮我分析这个图片里组件传递的功能语义，功能语义描述了这个组件的抽象功能，例如展示数据、描述数据变化趋势等，请用json方式返回格式化的数据。例如，环形图通过环形的比例来比较不同类别的数据份额，直观地展示各部分占整体的比例关系，帮助用户快速感知不同类别的相对大小；环形的设计使中心区域可用于显示汇总数值，增强信息的整体理解；颜色的区分进一步加强对数据特征及其状态的认知，提高图表的可读性,示例如下：
{
  "组件名": "环形图组件",
  "功能": [
    {
      "描述": "展示数据的整体结构和各部分的占比",
      "细节": [
        "通过环形的比例来比较不同类别的数据份额",
        "帮助用户快速感知不同类别的相对大小"
      ]
    },
    {
      "描述": "在中心区域突出显示汇总或关键数值",
      "细节": [
        "环形图中心可放置总量或其他关键信息",
        "增强用户对整体数值的认知和理解"
      ]
    },
    {
      "描述": "通过颜色区分不同数据类别或状态",
      "细节": [
        "使用不同颜色来表示不同类别或状态",
        "提高图表可读性并便于快速对比"
      ]
    }
  ]
}
'''
system_prompt = '你是一个智能助手，请根据用户输入的组件功能语义，生成对应的组件描述。'