import React from "react";
import { type LegendItem } from "../../types";
interface DefaultLegendProps {
    items: LegendItem[];
    className?: string;
    yAxisLabel?: React.ReactNode;
    xAxisLabel?: React.ReactNode;
    containerWidth?: number;
    isExpanded: boolean;
    setIsExpanded: (isExpanded: boolean) => void;
    style?: React.CSSProperties;
}
declare const DefaultLegend: React.NamedExoticComponent<DefaultLegendProps & React.RefAttributes<HTMLDivElement>>;
export { DefaultLegend };
export type { DefaultLegendProps };
//# sourceMappingURL=DefaultLegend.d.ts.map