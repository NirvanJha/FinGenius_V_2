import React from "react";
interface LabelTooltipProviderProps {
    children: React.ReactNode;
    delayDuration?: number;
    skipDelayDuration?: number;
    disableHoverableContent?: boolean;
}
interface LabelTooltipProps {
    children: React.ReactNode;
    content: string;
    side?: "top" | "bottom" | "left" | "right";
    sideOffset?: number;
    delayDuration?: number;
    className?: string;
    disabled?: boolean;
    open?: boolean;
    defaultOpen?: boolean;
    onOpenChange?: (open: boolean) => void;
}
declare const LabelTooltipProvider: React.FC<LabelTooltipProviderProps>;
declare const LabelTooltip: React.ForwardRefExoticComponent<LabelTooltipProps & React.RefAttributes<HTMLDivElement>>;
export { LabelTooltip, LabelTooltipProvider };
export type { LabelTooltipProps, LabelTooltipProviderProps };
//# sourceMappingURL=LabelTooltip.d.ts.map