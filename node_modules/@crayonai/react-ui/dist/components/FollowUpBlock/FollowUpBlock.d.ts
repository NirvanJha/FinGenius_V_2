import { CSSProperties, ReactElement } from "react";
import { FollowUpItemProps } from "../FollowUpItem";
export interface FollowUpBlockProps extends React.HTMLAttributes<HTMLDivElement> {
    children: ReactElement<FollowUpItemProps> | ReactElement<FollowUpItemProps>[];
    className?: string;
    style?: CSSProperties;
}
declare const FollowUpBlock: import("react").ForwardRefExoticComponent<FollowUpBlockProps & import("react").RefAttributes<HTMLDivElement>>;
export { FollowUpBlock };
//# sourceMappingURL=FollowUpBlock.d.ts.map