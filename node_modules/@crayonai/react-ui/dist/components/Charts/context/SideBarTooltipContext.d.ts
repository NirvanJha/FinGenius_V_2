import React, { ReactNode } from "react";
export interface SideBarChartData {
    title: string;
    values: {
        value: number;
        label: string;
        color: string;
    }[];
}
interface SideBarTooltipContextType {
    data: SideBarChartData | undefined;
    isSideBarTooltipOpen: boolean;
    setData: (data: SideBarChartData) => void;
    setIsSideBarTooltipOpen: (isOpen: boolean) => void;
}
declare const SideBarTooltipContext: React.Context<SideBarTooltipContextType | undefined>;
interface SideBarTooltipProviderProps {
    children: ReactNode;
    isSideBarTooltipOpen: boolean;
    setIsSideBarTooltipOpen: (isOpen: boolean) => void;
    data: SideBarChartData | undefined;
    setData: (data: SideBarChartData) => void;
}
export declare const SideBarTooltipProvider: React.FC<SideBarTooltipProviderProps>;
export declare const useSideBarTooltip: () => SideBarTooltipContextType;
export default SideBarTooltipContext;
//# sourceMappingURL=SideBarTooltipContext.d.ts.map