import { jsx as _jsx } from "react/jsx-runtime";
import { createContext, useContext } from "react";
const SideBarTooltipContext = createContext(undefined);
export const SideBarTooltipProvider = ({ children, isSideBarTooltipOpen, setIsSideBarTooltipOpen, data, setData, }) => {
    const value = {
        data,
        isSideBarTooltipOpen,
        setData,
        setIsSideBarTooltipOpen,
    };
    return _jsx(SideBarTooltipContext.Provider, { value: value, children: children });
};
export const useSideBarTooltip = () => {
    const context = useContext(SideBarTooltipContext);
    if (context === undefined) {
        throw new Error("useSideBarTooltip must be used within a SideBarTooltipProvider");
    }
    return context;
};
export default SideBarTooltipContext;
//# sourceMappingURL=SideBarTooltipContext.js.map