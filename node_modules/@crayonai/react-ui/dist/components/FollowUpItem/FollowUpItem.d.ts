import { ButtonHTMLAttributes, ReactNode } from "react";
export interface FollowUpItemProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    text: ReactNode;
    icon?: ReactNode;
    className?: string;
}
declare const FollowUpItem: import("react").ForwardRefExoticComponent<FollowUpItemProps & import("react").RefAttributes<HTMLButtonElement>>;
export { FollowUpItem };
//# sourceMappingURL=FollowUpItem.d.ts.map