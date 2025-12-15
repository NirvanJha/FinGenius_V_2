import type { Placement } from "@floating-ui/react-dom";
import React from "react";
interface FloatingUIPortalProps {
    children: React.ReactNode;
    className?: string;
    chartId?: string;
    portalContainer?: React.RefObject<HTMLElement | null>;
    position?: Partial<{
        x: number;
        y: number;
    }>;
    placement?: Placement;
    offsetDistance?: number;
}
export declare const FloatingUIPortal: React.FC<FloatingUIPortalProps>;
export {};
//# sourceMappingURL=FloatingUIPortal.d.ts.map