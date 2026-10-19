import { CheckOutlined, CloseOutlined, FilePdfOutlined } from "@ant-design/icons";
import { Button, Input, Tag } from "antd";
import { useState } from "react";

export interface ApprovalAction {
  tool: string;
  args: Record<string, unknown>;
  description: string;
}

interface ApprovalPanelProps {
  actions: ApprovalAction[];
  isSubmitting: boolean;
  onSubmit: (decision: "approve" | "reject", feedback: string) => Promise<unknown>;
}

const TOOL_LABELS: Record<string, string> = {
  convert_md_to_pdf: "Markdown 转 PDF",
  generate_markdown: "生成 Markdown",
  read_file_content: "读取附件"
};

function describeArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args || {});
  if (entries.length === 0) {
    return "无参数";
  }
  return entries
    .map(([key, value]) => `${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`)
    .join("；");
}

export function ApprovalPanel({ actions, isSubmitting, onSubmit }: ApprovalPanelProps) {
  const [feedback, setFeedback] = useState("");
  const [mode, setMode] = useState<"approve" | "reject">("approve");

  async function handleSubmit() {
    if (mode === "reject" && !feedback.trim()) {
      return;
    }
    await onSubmit(mode, feedback.trim());
    setFeedback("");
  }

  return (
    <section className="approval-panel" aria-label="人工确认">
      <div className="approval-heading">
        <FilePdfOutlined aria-hidden />
        <div>
          <strong>高风险动作待确认</strong>
          <p>系统已暂停在交付前，确认后才会继续执行；拒绝时可填写修改意见。</p>
        </div>
      </div>

      <ul className="approval-actions">
        {actions.map((action, index) => (
          <li key={`${action.tool}-${index}`}>
            <Tag color="gold">{TOOL_LABELS[action.tool] || action.tool}</Tag>
            <span className="approval-args">{describeArgs(action.args)}</span>
          </li>
        ))}
      </ul>

      <Input.TextArea
        autoSize={{ minRows: 2, maxRows: 5 }}
        onChange={(event) => setFeedback(event.target.value)}
        onFocus={() => setMode("reject")}
        placeholder="需要调整报告结构或内容时，在这里填写修改意见（填写后按“拒绝并重写”提交）"
        value={feedback}
      />

      <div className="approval-buttons">
        <Button
          disabled={isSubmitting}
          icon={<CheckOutlined />}
          loading={isSubmitting && mode === "approve"}
          onClick={() => {
            setMode("approve");
            onSubmit("approve", "");
          }}
          type="primary"
        >
          确认大纲并生成 PDF
        </Button>
        <Button
          danger
          disabled={isSubmitting || !feedback.trim()}
          icon={<CloseOutlined />}
          loading={isSubmitting && mode === "reject"}
          onClick={handleSubmit}
        >
          拒绝并重写
        </Button>
      </div>
    </section>
  );
}
